# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import json
import os, copy
from typing import List, Optional, Union
from pathlib import Path

import hydra
import sacrebleu
import torch
from hydra.utils import get_class
from omegaconf import ListConfig
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.collections.asr.parts.mixins.transcription import move_to_device
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.asr.parts.utils.eval_utils import remove_punctuations
from nemo.collections.common.metrics import MetricStringToTorchMetric, TextMetricsSet
from nemo.collections.multimodal.speech_llm.data.audio_text_qa_dataset import (
    get_aqa_dataset_from_config,
    get_tarred_aqa_dataset_from_config,
)
from nemo.collections.multimodal.speech_llm.modules.common.audio_text_generation_utils import generate
from nemo.collections.multimodal.speech_llm.modules.perception import AudioPerceptionModule, MultiAudioPerceptionModule
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingBatchSampler,
)
from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    build_position_ids,
)
from nemo.collections.nlp.modules.common.text_generation_utils import get_computeprob_response
from nemo.collections.nlp.parts.peft_config import PEFT_CONFIG_MAP
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.classes.mixins import adapter_mixins
from nemo.utils import AppState, logging, model_utils

try:
    from apex.transformer.pipeline_parallel.utils import _reconfigure_microbatch_calculator, get_num_microbatches

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

try:
    from megatron.core import InferenceParams, parallel_state, tensor_parallel
    from megatron.core.models.gpt import GPTModel as MCoreGPTModel

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE = False


__all__ = ["ModularAudioGPTModel"]


default_inference_config = {'tokens_to_generate': 30}


class ModularAudioGPTModel(MegatronGPTSFTModel):
    """Modularized speech GPT model."""

    def setup_perception_modules(self, cfg):
        if 'target' in cfg.perception:
            imported_cls = model_utils.import_class_by_path(cfg.perception.target)
            pretrained_audio_model = cfg.pretrained_canary_model if hasattr(cfg, "pretrained_canary_model") else cfg.pretrained_audio_model
            self.perception = imported_cls(
                cfg=cfg.perception
            )
        else:
            self.perception = (
                AudioPerceptionModule(cfg=cfg.perception)
                if "encoders" not in cfg.perception
                else MultiAudioPerceptionModule(cfg=cfg.perception)
            )

        if self._cfg.get('tensor_model_parallel_size', 1) == 1:
            audio_model, _ = self.get_audio_encoder_models_and_configs(cfg)
            self.perception.tokenizer = audio_model.tokenizer

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        self.cfg = cfg
        super().__init__(cfg, trainer)
        self.setup_perception_modules(cfg)
        
        # print out params in more details
        self.summarize(max_depth=2)

    def parameters(self):
        # override the same method in MegatronGPT model to include parameters ouside of LM
        all_names = []
        all_params = []
        for name, param in self.named_parameters(recurse=True):
            all_names.append(name)
            all_params.append(param)

        if isinstance(self.model, list):
            for module in self.model:
                for name, param in module.named_parameters(recurse=True):
                    all_names.append(name)
                    all_params.append(param)

        return itertools.chain(all_params)

    def setup_optimizer_param_groups(self):
        """
        ModelPT override. Optimizer will get self._optimizer_param_groups. 
        Makes two optimizer param groups, one for the frozen model params
        and one for the prompt-table/prompt-encoder params. The learning 
        rate for the frozen model's params will always be zero effectively
        freezing the model's params but still allowing for the needed gradients
        to be passed around in pipeline parallel models. The prompt-encoder 
        and/or prompt table will use the learning rate set by the user. 
        """
        known_groups = []
        self.unfreeze()
        freeze_llm = self.cfg.get('freeze_llm', True)
        if freeze_llm:
            known_groups.append('model.')

        for param in self.model.parameters():
            param.requires_grad = not freeze_llm

        if self.cfg.get('freeze_audio_encoder', False):
            if self.cfg.perception.get("speaker_model", None) is not None:
                if self.cfg.perception.speaker_model.get("freeze", False):
                    self.perception.speaker_model.freeze()
                    known_groups.append('perception.speaker_model.')
            if self.cfg.perception.get("encoders", None) is not None:
                for key, enc_cfg in self.cfg.perception.encoders.items():
                    if enc_cfg.get("freeze", False):
                        self.perception.encoders[key].freeze()
                        known_groups.append(f'perception.encoders.{key}.')
            else:
                self.perception.encoder.freeze()
                known_groups.append('perception.encoder.')

        if self.cfg.get('freeze_modality_adapter', False):
            self.perception.modality_adapter.freeze()
            known_groups.append('perception.modality_adapter.')

        opt_params = []
        for _, module in self.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin) and module.is_adapter_available():
                module.set_enabled_adapters(enabled=True)
                module.unfreeze_enabled_adapters()  # selectively unfreeze the adapter modules.
                opt_params += [p for p in module.parameters()]

        param_groups = []
        if "optim_param_groups" in self.cfg:
            param_groups_cfg = self.cfg.optim_param_groups
            for group, group_cfg in param_groups_cfg.items():
                module = getattr(self, group, None)
                if module is None:
                    raise ValueError(f"{group} not found in model.")
                elif hasattr(module, "parameters"):
                    known_groups.append(f"{group}.")
                    new_group = {"params": module.parameters()}
                    for k, v in group_cfg.items():
                        new_group[k] = v
                    param_groups.append(new_group)
                else:
                    raise ValueError(f"{group} does not have parameters.")

        for n, p in self.named_parameters():
            is_unknown = True
            for group in known_groups:
                if n.startswith(group):
                    is_unknown = False
            if is_unknown:
                opt_params.append(p)

        param_groups = [{"params": opt_params}] + param_groups

        self._optimizer_param_groups = param_groups
        logging.info(f"Optimizer groups set:\n{self.summarize(max_depth=2)}")

    def _create_attention_mask(self, encoder_input: torch.Tensor):
        # Create causal attention mask for whole input
        batch_size = encoder_input.shape[0]
        max_len = encoder_input.shape[1]
        attention_mask = torch.tril(torch.ones((batch_size, max_len, max_len), device=encoder_input.device)).view(
            batch_size, 1, max_len, max_len
        )
        # Convert attention mask from float to bool
        attention_mask = attention_mask < 0.5
        return attention_mask

    def _concat_features(self, embs1, emb1_lens, embs2, emb2_lens):
        """Concatenate two sets of embeddings and their lengths."""
        concat_emb = []
        concat_len = []
        for emb1, emb1_len, emb2, emb2_len in zip(embs1, emb1_lens, embs2, emb2_lens):
            new_len = emb1_len + emb2_len
            new_emb = torch.concat([emb1[:emb1_len], emb2[:emb2_len]], axis=0)
            padded_new_emb = torch.zeros(emb1.shape[0] + emb2.shape[0], emb1.shape[-1], device=emb1.device)
            padded_new_emb[:new_len, ...] = new_emb
            concat_emb.append(padded_new_emb)
            concat_len.append(new_len)
        concat_emb = torch.stack(concat_emb, dim=0)
        concat_len = torch.stack(concat_len, dim=0)
        return concat_emb, concat_len

    def _concat_multi_features(
        self,
        encoded: List[torch.Tensor],
        encoded_len: List[torch.Tensor],
        input_embeds: torch.Tensor,
        input_length: torch.Tensor,
        context_start_idx: List[List[int]],
    ):
        """Concatenate multiple audio features with text segments."""
        encoder_input_list, encoder_length_list = [], []
        batch_size = input_embeds.size(0)
        max_length = 0
        for i in range(batch_size):
            start_idx_list_i = context_start_idx[i] + [
                input_embeds.size(1)
            ]  # use input_embeds instead of input_length to handle tokens_to_generate in inference
            input_len_list = [start_idx_list_i[j + 1] - start_idx_list_i[j] for j in range(len(start_idx_list_i) - 1)]
            input_emb_list = input_embeds[i].split(input_len_list)
            encoder_input_i = [input_emb_list[0]]
            for j in range(1, len(input_emb_list)):
                encoder_input_i.append(encoded[i][j - 1][: encoded_len[i][j - 1]])
                encoder_input_i.append(input_emb_list[j])
            encoder_input_i = torch.cat(encoder_input_i)  # T, C
            encoder_length_i = encoded_len[i].sum() + input_length[i]  # total length of audio and text features
            max_length = max(max_length, encoder_input_i.size(0))
            encoder_input_list.append(encoder_input_i)
            encoder_length_list.append(encoder_length_i)

        encoder_input = torch.stack(
            [torch.nn.functional.pad(f, (0, 0, 0, max_length - f.size(0))) for f in encoder_input_list]
        )
        encoder_length = torch.LongTensor(encoder_length_list).to(encoder_input.device)
        return encoder_input, encoder_length

    def inject_perception_input(
        self,
        encoded: Union[torch.Tensor, List[torch.Tensor]],
        encoded_len: Union[torch.Tensor, List[torch.Tensor]],
        input_ids: torch.Tensor,
        input_length: torch.Tensor,
        context_start_idx: Optional[List[List[int]]] = None,
    ):
        """Inject audio features into the text input and return the final input embeddings to LLM."""
        # [b, t, c]
        lm_embedding = (
            self.model.language_model.embedding if hasattr(self.model, 'language_model') else self.model.embedding
        )
        input_embeds = lm_embedding.word_embeddings(input_ids)
        if isinstance(encoded, torch.Tensor):
            # single audio
            encoder_input, encoder_length = self._concat_features(encoded, encoded_len, input_embeds, input_length)
        else:
            # concat multiple audios with text segments
            encoder_input, encoder_length = self._concat_multi_features(
                encoded, encoded_len, input_embeds, input_length, context_start_idx
            )

        attention_mask = self._create_attention_mask(encoder_input)
        position_ids = build_position_ids(encoder_input[:, :, 0])

        # Add position embeddings
        if (
            getattr(lm_embedding, "position_embeddings", None) is not None
            and lm_embedding.position_embedding_type == 'learned_absolute'
        ):
            position_embeddings = lm_embedding.position_embeddings(position_ids)
            encoder_input = encoder_input + position_embeddings

        encoder_max_length = encoder_input.shape[1]
        if not hasattr(lm_embedding, 'transpose_batch_sequence') or lm_embedding.transpose_batch_sequence:
            encoder_input = encoder_input.transpose(0, 1).contiguous()
        if self.cfg.get("sequence_parallel", False):
            encoder_input = tensor_parallel.mappings.scatter_to_sequence_parallel_region(encoder_input)
        return encoder_input, attention_mask, encoder_length, position_ids, encoder_max_length

    def _shift_labels_by_emb_len(self, labels, label_lens, emb_lens, max_len, pad_token=0):
        """Shift labels to the right by the length of the audio embeddings."""
        shifted_labels = []
        for label, label_len, emb_len in zip(labels, label_lens, emb_lens):
            shifted_label = torch.full([max_len], pad_token, device=label.device)
            shifted_label[emb_len : emb_len + label_len] = label[:label_len]
            shifted_labels.append(shifted_label)
        shifted_labels = torch.stack(shifted_labels, dim=0)
        return shifted_labels

    def _get_text_embeddings(self, text_tokens, position_ids):
        """Get text embeddings for the input text tokens."""
        lm_embedding = (
            self.model.language_model.embedding if hasattr(self.model, 'language_model') else self.model.embedding
        )
        text_embeddings = lm_embedding.word_embeddings(text_tokens)  # (batch_size, seq_len, hidden_size)
        if hasattr(lm_embedding, 'position_embeddings'):
            position_embeddings = lm_embedding.position_embeddings(position_ids)
            text_embeddings = text_embeddings + position_embeddings
        return text_embeddings.transpose(0, 1)

    def prepare_llm_input(self, audio_batch):
        """Prepare input for the LLM."""
        input_signal = audio_batch['audio_signal']
        input_signal_length = audio_batch['audio_signal_length']

        input_ids, input_length, labels, loss_mask = (
            audio_batch['tokens'],
            audio_batch['tokens_length'],
            audio_batch['labels'],
            audio_batch['loss_mask'],
        )

        num_audios = audio_batch.get("num_audios", None)
        context_start_idx = audio_batch.get("context_start_idx", None)

        # [b, t, c]
        encoded, encoded_len = self.perception(
            input_signal=input_signal,
            input_signal_length=input_signal_length,
            processed_signal=None,
            processed_signal_length=None,
        )

        if num_audios is not None:
            # split the encoded and encoded_len by num_audios, used when there're multiple audio files per sample
            encoded = encoded.split(num_audios.tolist())
            encoded_len = encoded_len.split(num_audios.tolist())

        encoder_input, attention_mask, encoder_length, _, encoder_max_length = self.inject_perception_input(
            encoded, encoded_len, input_ids, input_length, context_start_idx
        )
        if num_audios is not None:
            # sum up the audio_feat_lens for each sample in the batch
            encoded_len = torch.stack([torch.sum(lens) for lens in encoded_len])

        # Shift labels to the right
        labels = self._shift_labels_by_emb_len(labels, input_length, encoded_len, encoder_max_length, pad_token=0)
        # Loss mask where answer tokens are 1.0 and all other tokens are 0.0
        loss_mask = self._shift_labels_by_emb_len(
            loss_mask, input_length, encoded_len, encoder_max_length, pad_token=0
        )

        return encoder_input, attention_mask, labels, loss_mask, encoder_length

    def forward(
        self, audio_batch, checkpoint_activations_all_layers,
    ):
        """
        Forward pass of the model. We prepend audio embeddings to the instruction and label text tokens as the LLM input.
        """
        if 'audio_ratio' in audio_batch:
            self.log(
                'local_batch_size',
                audio_batch['audio_ratio'].shape[0],
                prog_bar=True,
                batch_size=1,
                rank_zero_only=False,
            )

        encoder_input, attention_mask, labels, loss_mask, _ = self.prepare_llm_input(audio_batch)
        if self.mcore_gpt:
            output = self.model(
                input_ids=None,
                position_ids=None,
                decoder_input=encoder_input,
                attention_mask=attention_mask,
                labels=labels,
            )
        else:
            output = self.model(
                input_ids=None,
                position_ids=None,
                encoder_input=encoder_input,
                attention_mask=attention_mask,
                labels=labels,
                checkpoint_activations_all_layers=checkpoint_activations_all_layers,
            )

        return output, loss_mask

    def get_forward_output_only_func(self):
        def fwd_output_only_func(dataloader_iter, model):
            batch = next(dataloader_iter)
            extra_arg = {}
            # take the batch produced by prepare_batch_at_step
            (
                tokens,
                input_embeddings,
                attention_mask,
                position_ids,
                set_inference_key_value_memory,
                inference_max_sequence_len,
            ) = batch
            tokens = tokens.cuda()
            position_ids = position_ids.cuda()
            if attention_mask is not None:
                attention_mask = attention_mask.cuda()
                attention_mask = attention_mask[0:1]
            if self.mcore_gpt:
                # if first step, then clear KV cache, otherwise reuse inference_paarms
                if set_inference_key_value_memory[0].item():
                    self.inference_params = InferenceParams(
                        max_batch_size=tokens.size(0), max_sequence_length=inference_max_sequence_len[0].item()
                    )
                extra_arg['inference_params'] = self.inference_params
            else:
                extra_arg['set_inference_key_value_memory'] = set_inference_key_value_memory[0].item()
                extra_arg['inference_max_sequence_len'] = inference_max_sequence_len[0].item()

            # Currently for all MCore transformer layer specs causal attention mask
            # is used so we can delegate creating it to MCore/TE and pass None below
            if (
                isinstance(model, MCoreGPTModel)
                or hasattr(model, "module")
                and isinstance(model.module, MCoreGPTModel)
            ):
                attention_mask = None

            output_tensor = model(
                input_ids=None,
                position_ids=None,
                decoder_input=input_embeddings,
                attention_mask=attention_mask,
                **extra_arg,
            )

            # Advance inference sequence offset.
            if self.inference_params:
                # if last stage, then (final) output is [b, s, h], otherwise it's [s, b, h]
                if parallel_state.is_pipeline_last_stage():
                    self.inference_params.sequence_len_offset += output_tensor.size(1)
                else:
                    self.inference_params.sequence_len_offset += output_tensor.size(0)

            def id_func(output_tensor):
                return output_tensor, {'logits': output_tensor}

            return output_tensor, id_func

        return fwd_output_only_func

    def get_forward_output_and_loss_func(self, validation_step=False, tuning=False):
        def fwd_output_and_loss_func(dataloader_iter, model, checkpoint_activations_all_layers=None):
            batch = next(dataloader_iter)

            # Transfer needed data to GPU
            required_keys = set()
            if parallel_state.get_pipeline_model_parallel_world_size() == 1:
                required_keys.update(batch.keys())
            else:
                required_keys.add('attention_mask')
                if parallel_state.is_pipeline_first_stage():
                    required_keys.update(('tokens', 'position_ids'))
                if parallel_state.is_pipeline_last_stage():
                    required_keys.update(('labels', 'loss_mask'))
            if self.get_attention_mask_from_fusion and 'attention_mask' in required_keys:
                required_keys.remove('attention_mask')

            batch = move_to_device(batch, self.device)
            batch = self.get_batch_on_this_context_parallel_rank(batch)

            if not self.mcore_gpt:
                batch['checkpoint_activations_all_layers'] = checkpoint_activations_all_layers

            output_tensor, loss_mask = self.forward(
                batch, checkpoint_activations_all_layers=checkpoint_activations_all_layers
            )
            batch['loss_mask'] = loss_mask

            def loss_func(output_tensor):
                # Loss for a micro-batch (ub)
                loss_for_ub = self.loss_func(batch['loss_mask'], batch['num_valid_tokens_in_ub'], output_tensor)
                cp_size = self.cfg.get('context_parallel_size', 1)
                if self.cfg.data.get(
                    "return_output_tensors", False
                ):  # TODO: need a better way to check if loss_func is returning more stuff than just loss... (@adithyare)
                    loss_for_ub, q_hs, d_hs, pos_cs, neg_cs, diff_cs = loss_for_ub
                    reduced_loss = average_losses_across_data_parallel_group([loss_for_ub])
                    pos_cs = average_losses_across_data_parallel_group([pos_cs])
                    neg_cs = average_losses_across_data_parallel_group([neg_cs])
                    diff_cs = average_losses_across_data_parallel_group([diff_cs])
                    return (
                        loss_for_ub * cp_size,
                        {
                            'avg': reduced_loss,
                            'query_hs': q_hs,
                            'doc_hs': d_hs,
                            'avg_pos_cs': pos_cs,
                            'avg_neg_cs': neg_cs,
                            'diff_cs': diff_cs,
                        },
                    )
                elif validation_step and not self.cfg.data.get('validation_drop_last', True):
                    num_valid_tokens_in_ub = batch['num_valid_tokens_in_ub']
                    if loss_for_ub.isnan():
                        assert batch['loss_mask'].count_nonzero() == 0, 'Got NaN loss with non-empty input'
                        loss_sum_for_ub = torch.zeros_like(num_valid_tokens_in_ub)
                    else:
                        loss_sum_for_ub = num_valid_tokens_in_ub * loss_for_ub

                    loss_sum_and_ub_size_all_gpu = torch.cat(
                        [
                            loss_sum_for_ub.clone().detach().view(1),
                            torch.tensor([num_valid_tokens_in_ub]).cuda().clone().detach(),
                        ]
                    )
                    # Could potentially reduce num_valid_samples_in_microbatch and use that to aggregate instead of len(self._validation_ds)
                    torch.distributed.all_reduce(
                        loss_sum_and_ub_size_all_gpu, group=parallel_state.get_data_parallel_group()
                    )
                    return loss_for_ub * cp_size, {'loss_sum_and_ub_size': loss_sum_and_ub_size_all_gpu}
                else:
                    reduced_loss = average_losses_across_data_parallel_group([loss_for_ub])
                    return loss_for_ub * cp_size, {'avg': reduced_loss}

            return output_tensor, loss_func

        return fwd_output_and_loss_func

    def _build_dataset(self, data_cfg, is_train=True):
        if 'augmentor' in data_cfg:
            augmentor = process_augmentations(
                data_cfg['augmentor'], global_rank=self.global_rank, world_size=self.world_size
            )
        else:
            augmentor = None

        # Check dataset max_seq_legnth and max_position_embeddings size
        if (
            self.cfg.get('position_embedding_type', None) in [None, 'learned_absolute']
            and data_cfg.max_seq_length > self.cfg.max_position_embeddings
        ):
            logging.warning(
                f"Set dataset max_seq_length to max_position_embeddings {self.cfg.max_position_embeddings} if using learned_absolute position embedding"
            )
            data_cfg.max_seq_length = self.cfg.max_position_embeddings

        # Notably, the data weights are controlled by either bucketing_weights
        # or concat_sampling_probabilities depending on the dataset type.
        if data_cfg.get("use_lhotse"):
            from nemo.collections.multimodal.speech_llm.data.lhotse_dataset import (
                LhotseAudioQuestionAnswerDataset,
                TextProcessing,
            )

            tp = TextProcessing(
                self.tokenizer,
                max_seq_length=data_cfg["max_seq_length"],
                min_seq_length=data_cfg["min_seq_length"],
                add_bos=data_cfg.get('add_bos', False),
                add_eos=data_cfg.get('add_eos', False),
                add_sep=data_cfg.get('add_sep', False),
                sep_id=self.sep_id,
                seed=data_cfg.get('seed', 1234),
                separate_prompt_and_response_with_newline=data_cfg.get(
                    'separate_prompt_and_response_with_newline', True
                ),
                answer_only_loss=self.cfg.get('answer_only_loss', True),
                truncation_field=data_cfg.get('truncation_field', 'context'),
                pad_to_max_length=data_cfg.get('pad_to_max_length', False),
                prompt_template=data_cfg.get('prompt_template', None),
                virtual_tokens=self.virtual_tokens,
                tokens_to_generate=data_cfg.get(
                    'tokens_to_generate', 0
                ),  # used at inference time to allocate tensor positions for tokens that will be generated by inf procedure.
                input_key=data_cfg.get('input_key', 'input'),
                output_key=data_cfg.get('output_key', 'output'),
                end_string=data_cfg.get('end_string', None),
                sample_alpha=data_cfg.get('sample_alpha', None),
                input_text_mask_ratio=data_cfg.get('input_text_mask_ratio', None),
            )
            if self.cfg.perception.get("is_canary", False):
                from nemo.collections.asr.data.audio_to_text_lhotse_prompted import (
                    PromptedAudioToTextLhotseDataset,
                    get_prompt_format_fn,
                )

                perception_tokenizer = (
                    self.perception.tokenizer
                    if hasattr(self.perception, "tokenizer")
                    else self.perception.asr_model.tokenizer
                )
                canary_processer=PromptedAudioToTextLhotseDataset(
                    tokenizer=perception_tokenizer,
                    prompt_format_fn=get_prompt_format_fn('canary'),
                    inference=True,
                )
            else:
                canary_processer = None
            return LhotseAudioQuestionAnswerDataset(
                tp,
                default_question="answer the question according to the previous audio",
                tokens_to_generate=data_cfg.get('tokens_to_generate', 0),
                pad_to_max_length=data_cfg.get('pad_to_max_length', False),
                max_seq_length=data_cfg["max_seq_length"],
                canary_processor=canary_processer,
                convert_canary_prompt_to_text=data_cfg.get('convert_canary_prompt_to_text', False),
                prepend_to_exist_question=data_cfg.get('prepend_to_exist_question', None),
                canary_tokens_augment_ratio=data_cfg.get('canary_tokens_augment_ratio', 0.0),
                random_context_prob=data_cfg.get('random_context_prob', 0.0),
                random_context_positive_percent=data_cfg.get('random_context_positive_percent', 0.1),
            )

        # Notably, the data weights are controlled by either bucketing_weights
        # or concat_sampling_probabilities depending on the dataset type.
        if data_cfg.get('is_tarred', False):
            return get_tarred_aqa_dataset_from_config(
                config=data_cfg,
                tokenizer=self.tokenizer,
                augmentor=augmentor,
                sep_id=self.sep_id,
                answer_only_loss=self.cfg.get('answer_only_loss', True),
                virtual_tokens=self.virtual_tokens,
                global_rank=parallel_state.get_data_parallel_rank(),
                world_size=parallel_state.get_data_parallel_world_size(),
            )
        else:
            return get_aqa_dataset_from_config(
                manifest_filepath=data_cfg.manifest_filepath,
                config=data_cfg,
                tokenizer=self.tokenizer,
                augmentor=augmentor,
                is_train=is_train,
                sep_id=self.sep_id,
                answer_only_loss=self.cfg.get('answer_only_loss', True),
                virtual_tokens=self.virtual_tokens,
            )

    def build_data_loader(self, dataset, data_cfg, consumed_samples=0, is_eval=False):
        """Buld dataloader given an input dataset."""
        if data_cfg.get("use_lhotse"):
            from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config

            if data_cfg.get('is_tarred', False) or not is_eval:
                return get_lhotse_dataloader_from_config(
                    data_cfg,
                    global_rank=parallel_state.get_data_parallel_rank(),
                    world_size=parallel_state.get_data_parallel_world_size(),
                    dataset=dataset,
                )
            else:
                dls = []
                for dataset_idx, (cur_manifest_filepath) in enumerate(data_cfg.manifest_filepath):
                    conf = copy.deepcopy(data_cfg)
                    conf['manifest_filepath'] = cur_manifest_filepath
                    question_file_set = data_cfg.get('question_file_set', None)
                    if question_file_set is not None:
                        conf['question_file_set'] = [question_file_set[dataset_idx]]
                    dls.append(
                        get_lhotse_dataloader_from_config(
                            conf,
                            global_rank=parallel_state.get_data_parallel_rank(),
                            world_size=parallel_state.get_data_parallel_world_size(),
                            dataset=dataset,
                        )
                    )
                if 'names' not in data_cfg:
                    names = []
                    for cur_manifest_filepath in data_cfg.manifest_filepath:
                        names.append(Path(cur_manifest_filepath).stem)
                    OmegaConf.update(data_cfg, 'names', names, force_add=True)
                    logging.info(f'Update dataset names as {names}')
                return dls

        logging.info(f'Building dataloader with consumed samples: {consumed_samples}')
        if isinstance(dataset, BlendableDataset):
            collate_fn = dataset.datasets[0].collate_fn
        elif hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        elif hasattr(dataset.datasets[0], 'collate_fn'):
            # support datasets that are lists of entries
            collate_fn = dataset.datasets[0].collate_fn
        else:
            # support datasets that are lists of lists
            collate_fn = dataset.datasets[0].datasets[0].collate_fn

        if isinstance(dataset, torch.utils.data.IterableDataset):
            data_parallel_size = parallel_state.get_data_parallel_world_size()
            num_micro_batches = data_cfg.global_batch_size // (data_cfg.micro_batch_size * data_parallel_size)
            global_batch_size_on_this_data_parallel_rank = num_micro_batches * data_cfg.micro_batch_size

            dataloader = torch.utils.data.DataLoader(
                dataset,
                collate_fn=collate_fn,
                shuffle=False,
                batch_size=global_batch_size_on_this_data_parallel_rank,
                drop_last=True,
                num_workers=data_cfg.num_workers,
                pin_memory=data_cfg.pin_memory,
            )
            return dataloader

        batch_sampler = MegatronPretrainingBatchSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=data_cfg.micro_batch_size,
            global_batch_size=data_cfg.global_batch_size,
            data_parallel_rank=parallel_state.get_data_parallel_rank(),
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
            drop_last=True,
            pad_samples_to_global_batch_size=False,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=data_cfg.num_workers,
            pin_memory=data_cfg.pin_memory,
        )
        return dataloader

    @classmethod
    def _modify_audio_encoder_config(cls, gpt_cfg, audio_cfg, speaker_cfg=None):
        with open_dict(gpt_cfg):
            use_multi_encoder = gpt_cfg.perception.get("encoders", None) is not None
            if not use_multi_encoder:
                gpt_cfg.perception.preprocessor = audio_cfg.preprocessor
                gpt_cfg.perception.encoder = audio_cfg.encoder
            else:
                for key in gpt_cfg.perception.encoders:
                    model_key = gpt_cfg.perception.encoders[key].get("model_key", "encoder")
                    gpt_cfg.perception.encoders[key]["model"] = audio_cfg[key][model_key]
                    if "preprocessor" in audio_cfg[key]:
                        gpt_cfg.perception.encoders[key]['preprocessor'] = audio_cfg[key].preprocessor
                if speaker_cfg is not None:
                    gpt_cfg.perception.speaker_model.model = speaker_cfg

            gpt_cfg.perception.output_dim = gpt_cfg.hidden_size
            modality_adapter_cfg = gpt_cfg.perception.modality_adapter
            if 'output_dim' in modality_adapter_cfg:
                modality_adapter_cfg.output_dim = gpt_cfg.hidden_size
            if not use_multi_encoder:
                if 'feat_in' in modality_adapter_cfg:  # conformer encoder
                    modality_adapter_cfg.feat_in = audio_cfg.encoder.d_model
                if 'input_dim' in modality_adapter_cfg:
                    modality_adapter_cfg.input_dim = audio_cfg.encoder.d_model

    @classmethod
    def _modify_config(cls, gpt_cfg, cfg, audio_cfg, add_cfg_to_tree=False, speaker_cfg=None):
        """
        This function modifies the original gpt pre-training config (gpt_cfg) with attributes from the finetuning config (cfg).
        The `add_cfg_to_tree` arg adds `cfg` to the top of the yaml tree which is needed for all `hparams.yaml` files when passed as an arg to `load_from_checkpoint()`.
        """
        OmegaConf.set_struct(gpt_cfg, True)
        OmegaConf.resolve(cfg)
        with open_dict(gpt_cfg):
            # for AudioGPTLoRAModel
            gpt_cfg.target = f"{cls.__module__}.{cls.__name__}"
            gpt_cfg.perception = cfg.model.perception
            cls._modify_audio_encoder_config(gpt_cfg, audio_cfg, speaker_cfg)
            # This is needed when modifying a hparam file directly to load `.ckpt` files.
            # This is not needed to modify the cfg in `.nemo` files.
            if add_cfg_to_tree:
                OmegaConf.resolve(gpt_cfg)
                gpt_cfg.cfg = gpt_cfg

        return gpt_cfg

    @classmethod
    def get_pretraind_audio_model(cls, encoder_cfg: DictConfig) -> ModelPT:
        encoder_cls = get_class(encoder_cfg.get("_target_")) if encoder_cfg.get("_target_", None) is not None else None
        pretrained_model = encoder_cfg.get('pretrained_model', None)
        if pretrained_model is None:
            return None
        if encoder_cls is None:
            raise ValueError(
                f"Must specify a valid encoder class in the via the `_target_` field in the config: {encoder_cfg}"
            )

        if pretrained_model.endswith('.nemo'):
            logging.info(f'Loading pretrained audio model from local file: {pretrained_model}')
            audio_model = encoder_cls.restore_from(pretrained_model, map_location='cpu')
        else:
            logging.info(f'Loading pretrained audio model from NGC: {pretrained_model}')
            audio_model = encoder_cls.from_pretrained(pretrained_model, map_location='cpu')
        return audio_model

    @classmethod
    def get_speaker_model_and_config(cls, cfg):
        if 'speaker_model' in cfg.model.perception:
            speaker_cfg = cfg.model.perception.speaker_model
            if speaker_cfg.get('pretrained_model', None) is not None:
                if speaker_cfg.pretrained_model.endswith('.nemo'):
                    logging.info(f'Loading pretrained speaker model from local file: {speaker_cfg.pretrained_model}')
                    speaker_model = EncDecSpeakerLabelModel.restore_from(
                        speaker_cfg.pretrained_model, map_location='cpu'
                    )
                else:
                    logging.info(f'Loading pretrained speaker model from NGC: {speaker_cfg.pretrained_model}')
                    speaker_model = EncDecSpeakerLabelModel.from_pretrained(
                        speaker_cfg.pretrained_model, map_location='cpu'
                    )
                return speaker_model, speaker_model.cfg
            return None, None
        else:
            return None, None

    @classmethod
    def get_audio_encoder_models_and_configs(cls, cfg):
        if 'encoders' in cfg.perception:
            audio_encoders = {}
            audio_enc_cfgs = {}
            for key, encoder_cfg in cfg.perception.encoders.items():
                audio_encoders[key] = cls.get_pretraind_audio_model(encoder_cfg)
                audio_enc_cfgs[key] = audio_encoders[key].cfg
            return audio_encoders, audio_enc_cfgs
        else:
            pretrained_audio_model = cfg.get("pretrained_audio_model", None)
            pretrained_audio_model_class = cfg.get(
                "pretrained_audio_model_target", "nemo.collections.asr.models.ASRModel"
            )

            model_class = hydra.utils.get_class(pretrained_audio_model_class)
            if pretrained_audio_model.endswith('.nemo'):
                logging.info(f'Loading pretrained audio model from local file: {pretrained_audio_model}')
                audio_model = model_class.restore_from(pretrained_audio_model, map_location='cpu')
            else:
                logging.info(f'Loading pretrained audio model from NGC: {pretrained_audio_model}')
                audio_model = model_class.from_pretrained(pretrained_audio_model, map_location='cpu')
            return audio_model, audio_model.cfg

    @classmethod
    def load_pretrained_audio_weights(
        cls, cfg, model, audio_model, speaker_model: Optional[EncDecSpeakerLabelModel] = None
    ):
        model.perception.tokenizer = audio_model.tokenizer
        use_multi_encoder = cfg.model.perception.get("encoders", None) is not None
        if not use_multi_encoder:
            if cfg.model.perception.get("use_multi_layer_feat", False):
                model.perception.encoder.encoder.load_state_dict(audio_model.encoder.state_dict(), strict=True)
            else:
                model.perception.encoder.load_state_dict(audio_model.encoder.state_dict(), strict=True)
            logging.info(f'Loaded pretrained audio model weights from {cfg.model.pretrained_audio_model}')
            if cfg.model.get('use_am_tokenizer', False):
                model.tokenizer = audio_model.tokenizer
                logging.info(f'Use AM tokenizer: {audio_model.tokenizer}')
            return model
        else:
            for key, enc_cfg in cfg.model.perception.encoders.items():
                if enc_cfg.get("use_multi_layer_feat", False):
                    model.perception.encoders[key].encoder.load_state_dict(
                        audio_model[key].encoder.state_dict(), strict=True
                    )
                else:
                    model.perception.encoders[key].load_state_dict(audio_model[key].encoder.state_dict(), strict=True)
                logging.info(f'Loaded pretrained audio model weights for {key}')
            if speaker_model is not None:
                model.perception.speaker_model.load_state_dict(speaker_model.state_dict(), strict=True)
                logging.info(f'Loaded pretrained speaker model weights')
            return model

    @classmethod
    def restore_from_pretrained_models(
        cls, cfg: Optional[Union[OmegaConf, str]] = None, trainer: Optional[Trainer] = None,
    ):
        if (
            cfg.model.get("pretrained_audio_model", None) is None
            and cfg.model.perception.get("encoders", None) is None
        ):
            raise RuntimeError("PEFT training needs at least one pretrained audio model present.")

        if not cfg.model.restore_from_path:
            raise RuntimeError("PEFT training needs a trained base model present.")

        base_model_cfg = MegatronGPTSFTModel.merge_cfg_with(cfg.model.restore_from_path, cfg)
        audio_model, audio_model_cfg = cls.get_audio_encoder_models_and_configs(cfg.model)
        speaker_model, speaker_cfg = cls.get_speaker_model_and_config(cfg)
        model_cfg = cls._modify_config(
            base_model_cfg, cfg, audio_model_cfg, add_cfg_to_tree=False, speaker_cfg=speaker_cfg
        )

        # load llm
        model = cls.restore_from(
            restore_path=cfg.model.restore_from_path, trainer=trainer, override_config_path=model_cfg, strict=False,
        )

        if "peft" in cfg.model:
            peft_cfg_cls = PEFT_CONFIG_MAP[cfg.model.peft.peft_scheme]
            if cfg.model.peft.restore_from_path is not None:
                # initialize peft weights from a checkpoint instead of randomly
                # This is not the same as resume training because optimizer states are not restored.
                logging.info("PEFT Weights will be loaded from", cfg.model.peft.restore_from_path)
                model.load_adapters(cfg.model.peft.restore_from_path, peft_cfg_cls(model_cfg))
            elif peft_cfg_cls is not None:
                logging.info("Adding adapter weights to the model for PEFT")
                model.add_adapter(peft_cfg_cls(model_cfg))
            else:
                raise ValueError(f"PEFT scheme not not found in PEFT_CONFIG_MAP: {cfg.model.peft.peft_scheme}")
        else:
            logging.info(f"Running full finetuning since no peft scheme is given.\n{model.summarize()}")

        # load audio model weights
        model = cls.load_pretrained_audio_weights(cfg, model, audio_model, speaker_model)

        if 'inference' in cfg:
            inference_cfg = OmegaConf.to_container(cfg.inference, resolve=True)
            model.set_inference_config(inference_cfg)
        return model

    def _build_vocab(self):
        """
        Manipulate vocabulary (e.g., pad vocabulary for increased performance)/
        """
        if self._cfg.get('override_vocab_size', None) is not None:
            self.padded_vocab_size = self._cfg.override_vocab_size
        else:
            self.padded_vocab_size = self._vocab_size_with_padding(
                orig_vocab_size=self.tokenizer.vocab_size,
                make_vocab_size_divisible_by=self._cfg.get('make_vocab_size_divisible_by', 128),
                tensor_model_parallel_size=self._cfg.get('tensor_model_parallel_size', 1),
            )

    def state_dict(self, destination=None, prefix=None, keep_vars=False):
        if self.setup_complete:
            # Once setup is complete we only need adapter and perception model.
            if self.cfg.freeze_llm and self.cfg.get("peft", None) is not None:
                return_state_dict = self.get_peft_state_dict()
            elif not self.cfg.freeze_llm:
                return_state_dict = self.model.state_dict(prefix="model.")
            else:
                return_state_dict = {}

            state_dict = self.perception.state_dict(prefix="perception.")
            if self.cfg.freeze_audio_encoder:
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith("perception.encoder.")}

            return_state_dict.update(state_dict)
            state_dict = self.perception.state_dict(prefix="perception.")
            return_state_dict.update(state_dict)
            return return_state_dict
        else:
            # we want all the params with the same keys as calling self.state_dict()
            # but we can't call self.state_dict() here as it would be a recursive call.
            # so we call self.model.state_dict(prefix="model.") which will return all the keys and params same as calling self.state_dict()
            if not self.cfg.freeze_llm:
                return_state_dict = self.model.state_dict(prefix="model.")
            else:
                return_state_dict = {}
            state_dict = self.perception.state_dict(prefix="perception.")
            if self.cfg.freeze_audio_encoder:
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith("perception.encoder.")}
            return_state_dict.update(state_dict)
            return return_state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        if self.setup_complete:
            super(MegatronGPTSFTModel, self).load_state_dict(state_dict, strict=False)
        else:
            if self.cfg.get('override_vocab_size', False):
                exclude_list = [
                    "model.language_model.embedding.word_embeddings.weight",
                    "model.language_model.output_layer.weight",
                ]
            else:
                exclude_list = []
            state_dict = {k: v for k, v in state_dict.items() if k not in exclude_list}
            super(MegatronGPTSFTModel, self).load_state_dict(state_dict, strict=strict)

    def on_load_checkpoint(self, checkpoint) -> None:
        """LightningModule hook:
         https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-load-checkpoint
         """
        checkpoint_state_dict = checkpoint['state_dict']
        self.load_state_dict(checkpoint_state_dict, strict=False)

    def setup_metric(self, data_cfg):
        metric_name = "exact_string_match"
        if not hasattr(data_cfg, "metric"):
            metric = MetricStringToTorchMetric["exact_string_match"]
        else:
            if not hasattr(data_cfg.metric, "name"):
                raise ValueError("Metric name is not provided in the metric config.")
            if data_cfg.metric.name == "loss":
                return None, "loss"
            if data_cfg.metric.name not in MetricStringToTorchMetric:
                raise KeyError(
                    f"{data_cfg.metric.name} is not supported. List of supported metrics: {MetricStringToTorchMetric.keys()}"
                )
            if data_cfg.metric.name in self._metrics_require_string2category_map:
                if data_cfg.metric.average is None:
                    raise ValueError(
                        f"{data_cfg.metric.name} requires specifying whether you want to compute a micro or macro average. Found None."
                    )
            if (
                data_cfg.metric.get('labels_are_strings', False)
                and data_cfg.metric.name in self._metrics_require_string2category_map
            ):
                if data_cfg.metric.num_classes is None:
                    raise ValueError(
                        "Number of classes is not provided in the metric section within the data config. "
                        f"Please provide the number of classes in the data config to use the {data_cfg.metric.name} metric."
                    )
                if data_cfg.metric.get('class_labels', None) is None or not isinstance(
                    data_cfg.metric.get('class_labels', None), ListConfig
                ):
                    raise ValueError(
                        "Class labels are not provided properly in the metric section witnin the data config. "
                        f"Please provide the class labels as a list of strings in the data config to use the {data_cfg.metric.name} metric."
                    )
                if len(data_cfg.metric.get('class_labels', None)) != data_cfg.metric.num_classes:
                    raise ValueError(
                        f"Number of class labels {len(data_cfg.metric.get('class_labels', None))} does not match `num_classes` : {data_cfg.metric.num_classes}"
                    )

            metric_name = data_cfg.metric.name
            metric_cls = MetricStringToTorchMetric[metric_name]
            if metric_name not in TextMetricsSet:
                metric = [metric_cls(**data_cfg.metric)]
            else:
                metric = [metric_cls()]
        return metric, metric_name

    def inference_step(self, dataloader_iter, mode):
        batch, batch_idx, dataloader_idx = next(dataloader_iter)
        data_cfg = self.cfg.data.validation_ds if mode == 'validation' else self.cfg.data.test_ds
        self._reconfigure_and_process_inference_batch(batch, data_cfg)
        # Meta data from dataset
        metadata = batch.get('metadata', [{}] * len(batch['tokens']))
        loss = super(MegatronGPTSFTModel, self).validation_step(itertools.chain([batch]), dataloader_idx)

        # We need _inference_config to get generation params
        # add_BOS and tokens_to_generate are set in dataset
        if self.get_inference_config() is None:
            logging.warning(f'inference_config is not set. Use default: {default_inference_config}')
            self.set_inference_config(inference_config=default_inference_config)
        self._inference_config['add_BOS'] = data_cfg.add_bos
        self._inference_config['tokens_to_generate'] = data_cfg.get('tokens_to_generate')

        output = self.predict_step(batch, batch_idx, dataloader_idx)

        inputs_text = [self.tokenizer.ids_to_text(c.tolist()) for c in batch['contexts']]
        labels_text = [self.tokenizer.ids_to_text(a.tolist()) for a in batch['answers']]
        preds_text = [
            self.tokenizer.ids_to_text(t[l.item() :][: data_cfg.get('tokens_to_generate')])
            for t, l in zip(output['token_ids'], batch['context_lengths'])
        ]

        if data_cfg.get("end_string", None):
            # sometimes data_cfg.end_string != self.tokenizer.ids_to_text(self.tokenizer.text_to_ids(data_cfg.end_string))
            # for example when data_cfg.end_string = "<end>", the end_string_re will start with " ?? "
            end_string_re = self.tokenizer.ids_to_text(self.tokenizer.text_to_ids(data_cfg.end_string))
            preds_text_cleaned = []
            labels_text_cleaned = []
            for p, l in zip(preds_text, labels_text):
                # remove end_string from the end of the string
                for es in [end_string_re, data_cfg.end_string]:
                    if p.endswith(es):
                        p = p[: -len(es)].strip()
                    if l.endswith(es):
                        l = l[: -len(es)].strip()
                preds_text_cleaned.append(p)
                labels_text_cleaned.append(l)
            preds_text = preds_text_cleaned
            labels_text = labels_text_cleaned

        if data_cfg.get("remove_text_pc", False):
            preds_text = [remove_punctuations(p.lower(), data_cfg.get("punctuations", None)) for p in preds_text]
            labels_text = [remove_punctuations(l.lower(), data_cfg.get("punctuations", None)) for l in labels_text]

        if data_cfg.get("log_every_n_steps", None) is not None:
            if batch_idx % data_cfg.log_every_n_steps == 0:
                logging.info(f"Input: `{inputs_text[0]}`")
                logging.info(f"Label: `{labels_text[0]}`")
                logging.info(f"Pred: `{preds_text[0]}`")

        # if loss is nan, print the input, label and pred
        if loss.isnan():
            logging.info("++++++++++++++ NaN loss detected ++++++++++++++")
            for i in range(len(inputs_text)):
                logging.info(f"Input: `{inputs_text[i]}`")
                logging.info(f"Label: `{labels_text[i]}`")
                logging.info(f"Pred: `{preds_text[i]}`")
            logging.info("++++++++++++++++++++++++++++++++++++++++++++++++")

        outputs = {
            'loss': loss,
            'preds': preds_text,  # [str]
            'labels': labels_text,  # [str]
            'inputs': inputs_text,  # [str]
            'metadata': metadata,  # [dict]
        }

        if mode == 'validation':
            if len(self._validation_dl) > 1:
                # super().validation_step appends just loss to self.validation_step_outputs, replace the last appended loss with the outputs dict
                self.validation_step_outputs[dataloader_idx][-1] = outputs
            else:
                # super().validation_step appends just loss to self.validation_step_outputs, replace the last appended loss with the outputs dict
                self.validation_step_outputs[-1] = outputs
        else:
            if len(self._test_dl) > 1:
                self.test_step_outputs[dataloader_idx][-1] = outputs
            else:
                self.test_step_outputs[-1] = outputs
        return outputs

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: Optional[int] = None):
        inference_config = self.get_inference_config()
        if inference_config is not None:
            # need to overwrite some configuration, make it immutable
            inference_config = inference_config.copy()
        else:
            self.set_inference_config(inference_config=default_inference_config)
            logging.warning(f'inference_config is not set. Use default: {default_inference_config}')
            inference_config = self.get_inference_config()

        if self.cfg.data.get('end_string', None):
            inference_config['end_strings'] = [self.cfg.data.end_string]

        global_batch_size_per_gpu = batch['tokens'].size(0)
        num_micro_batches_before_decode = get_num_microbatches()

        compute_logprob = inference_config.get('compute_logprob', False)
        if compute_logprob:
            inference_config['inputs'] = batch
            inference_config['tokens_to_generate'] = 1
            inference_config['all_probs'] = True
            inference_config["add_BOS"] = False
            inference_config['greedy'] = True
            response = generate(self, **inference_config)
            response = get_computeprob_response(self.tokenizer, response, batch)
        else:
            # for megatron_gpt_eval.py
            if isinstance(batch, list):
                inference_config['inputs'] = batch
            elif 'num_audios' in batch:
                # peft_eval.py
                inference_config['inputs'] = (
                    batch['contexts'].cuda(),
                    batch['context_lengths'].cuda(),
                    batch['audio_signal'].cuda(),
                    batch['audio_signal_length'].cuda(),
                    batch['num_audios'].cuda(),
                    batch['context_start_idx'],
                )
            else:
                # peft_eval.py
                inference_config['inputs'] = (
                    batch['contexts'].cuda(),
                    batch['context_lengths'].cuda(),
                    batch['audio_signal'].cuda(),
                    batch['audio_signal_length'].cuda(),
                )
            response = generate(self, **inference_config)

        app_state = AppState()
        _reconfigure_microbatch_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=global_batch_size_per_gpu * parallel_state.get_data_parallel_world_size(),
            micro_batch_size=global_batch_size_per_gpu // num_micro_batches_before_decode,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )

        # add audio offsets to context lengths for properly decoding only the response
        batch['context_lengths'] = batch['context_lengths'].cuda() + response['audio_feat_lens']

        return response

    def inference_epoch_end(self, outputs, mode, data_cfg):
        # Parent class will handle logging of the loss.
        if not outputs or (all([not x for x in outputs])):
            return

        if isinstance(outputs[0], dict):
            outputs = [outputs]

        averaged_loss = []
        averaged_metric = []
        # Log metrics for each provided validation/test dataset.
        for dataloader_idx, output in enumerate(outputs):
            if len(output) == 0:
                logging.warning(f"Empty output for dataloader_idx: {dataloader_idx}")
                continue
            # Expand on_validation_epoch_end from parent class MegatronGPTModel as on_validation_epoch_end doesnt take outputs arg
            loss_vals = [x['loss'] for x in output]
            if parallel_state.is_pipeline_last_stage():
                # only the last pipeline parallel stages return loss with their batch size
                if self.cfg.data.get('validation_drop_last', True):
                    loss = torch.stack(loss_vals).mean()
                else:
                    # Compute the avg loss by total_loss across all samples / total number of samples
                    total_loss_and_total_samples = torch.vstack(loss_vals).sum(axis=0)
                    avg_loss = total_loss_and_total_samples[0] / total_loss_and_total_samples[1]
                    loss = avg_loss.type(torch.float32).cuda()
            else:
                loss = torch.tensor(0.0, dtype=torch.float32).cuda()

            # we can only log on one rank if it is rank zero so we broadcast from last rank
            torch.distributed.broadcast(loss, get_last_rank())

            self.log('val_loss', loss, prog_bar=True, rank_zero_only=True, batch_size=1, sync_dist=True)

            # Determine the key used to log the loss based on the user provided name of the dataset or the dataloader index.
            loss_log_key = self._determine_log_key(data_cfg, dataloader_idx, "loss", mode)
            self.log(loss_log_key, loss, batch_size=1)
            averaged_loss.append(loss)

            # Gather the outputs object from all data parallel ranks since we are using the DistributedSampler which splits data across DDP ranks.
            gathered_outputs = [None for _ in range(parallel_state.get_data_parallel_world_size())]
            torch.distributed.all_gather_object(
                gathered_outputs,
                [
                    {'preds': x['preds'], 'labels': x['labels'], 'inputs': x['inputs'], 'metadata': x['metadata']}
                    for x in output
                ],
                group=parallel_state.get_data_parallel_group(),
            )

            # Remove duplicate examples due to distributed sampler.
            inp_label_set = set()
            deduplicated_outputs = {
                'preds': [],
                'labels': [],
                'inputs': [],
                'metadata': [],
            }
            total_size = 0
            for rank in range(0, parallel_state.get_data_parallel_world_size()):
                for batch in gathered_outputs[rank]:
                    for pred, label, input, metadata in zip(
                        batch['preds'], batch['labels'], batch['inputs'], batch['metadata']
                    ):
                        key = input + label + str(metadata)
                        total_size += 1
                        if key not in inp_label_set:
                            inp_label_set.add(key)
                            deduplicated_outputs['preds'].append(pred)
                            deduplicated_outputs['labels'].append(label)
                            deduplicated_outputs['inputs'].append(input)
                            deduplicated_outputs['metadata'].append(metadata)

            # Compute metric score
            metric_name = self.val_metric_name if mode == 'validation' else self.test_metric_name
            metric_label_key = self.val_metric_label_key if mode == 'validation' else self.test_metric_label_key
            if metric_name != 'loss':
                metric_log_key = self._determine_log_key(data_cfg, dataloader_idx, metric_name, mode)
                metric_fn = self.val_metric[0] if mode == 'validation' else self.test_metric[0]
                if metric_label_key in deduplicated_outputs['metadata'][0]:
                    labels = [m[metric_label_key] for m in deduplicated_outputs['metadata']]
                else:
                    labels = deduplicated_outputs['labels']

                # sacrebleu.corpus_bleu is commonly used which does not share
                # the same interface as other metrics. We handle it separately.
                if metric_name == 'bleu':
                    metric_result = torch.Tensor(
                        [sacrebleu.corpus_bleu(deduplicated_outputs['preds'], [labels]).score]
                    ).to(self.device)
                else:
                    for pred, label in zip(deduplicated_outputs['preds'], labels):
                        _ = metric_fn(pred, label)

                    metric_result = metric_fn.compute()

                if metric_name == 'rouge':
                    for k, v in metric_result.items():
                        if 'fmeasure' in k:
                            self.log(metric_log_key + f'_{k}', v.item(), sync_dist=True, batch_size=1)
                            logging.info(f"{mode} {metric_name} {k}: {v.item()}")
                    metric_result = metric_result['rouge1_fmeasure']
                else:
                    self.log(metric_log_key, metric_result.item(), sync_dist=True, batch_size=1)
                    logging.info(f"{mode} {metric_name}: {metric_result.item()}")

                metric_fn.reset()
                averaged_metric.append(metric_result)

            # Write predictions to file
            if self.global_rank == 0 and data_cfg.get("write_predictions_to_file", False):
                logging.info(
                    f"Total deduplicated inference data size: {total_size} to {len(deduplicated_outputs['inputs'])}"
                )

                # Check if the user provided a prefix path to the file(s) they want to write.
                if not hasattr(data_cfg, "output_file_path_prefix") or data_cfg.output_file_path_prefix is None:
                    raise ValueError(
                        f"Cannot write predictions to file when output_file_path_prefix is not set or present in the yaml config file."
                    )
                filename_log_key = self._determine_log_key(data_cfg, dataloader_idx, None, mode)
                output_dir = data_cfg.get("output_dir", "./")
                self.write_predictions_to_file(
                    deduplicated_outputs, f"{data_cfg.output_file_path_prefix}_{filename_log_key}", output_dir
                )

            torch.distributed.barrier(group=parallel_state.get_data_parallel_group())
            outputs[dataloader_idx].clear()  # free memory

        # Logging of the averaged metrics:
        averaged_loss = sum(averaged_loss) / len(averaged_loss)
        averaged_metric = sum(averaged_metric) / len(averaged_metric) if len(averaged_metric) > 0 else None
        averaged_loss = averaged_loss.to(self.device)
        if averaged_metric is not None:
            averaged_metric = averaged_metric.to(self.device)

        # Handle case where metrics can be nan or inf. This can break checkpoint save/load.
        if averaged_metric is not None and (torch.isinf(averaged_metric) or torch.isnan(averaged_metric)):
            app_state = AppState()
            monitor_mode = app_state.checkpoint_callback_params.mode
            assert monitor_mode in ['min', 'max']
            averaged_metric = 0.0 if monitor_mode == 'max' else 1e5

        if mode == 'validation':
            self.log("validation_loss", averaged_loss, batch_size=1, sync_dist=True)
            if averaged_metric is not None:
                self.log(f"validation_{self.val_metric_name}", averaged_metric, sync_dist=True, batch_size=1)
        elif mode == 'test':
            self.log("test_loss", averaged_loss, batch_size=1, sync_dist=True)
            if averaged_metric is not None:
                self.log(f"test_{self.test_metric_name}", averaged_metric, sync_dist=True, batch_size=1)

        # Merge the functionality of previous on_inference_epoch_end() within inference_epoch_end() func here
        app_state = AppState()
        self._restore_activation_checkpointing_args()
        if hasattr(self, "_train_ds"):
            _reconfigure_microbatch_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=self.cfg.data.train_ds.global_batch_size,
                micro_batch_size=self.cfg.data.train_ds.micro_batch_size,
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )
        # When running `trainer.validate()`, the training dataset is not available.
        else:
            logging.warning('No training data found, reconfiguring microbatches based on validation batch sizes.')
            _reconfigure_microbatch_calculator(
                rank=app_state.global_rank,
                rampup_batch_size=None,
                global_batch_size=data_cfg.global_batch_size,
                micro_batch_size=data_cfg.micro_batch_size,
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )

        return averaged_loss, averaged_metric

    # consistent with speech models
    def write_predictions_to_file(self, outputs, output_file_path_prefix, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = output_file_path_prefix + "_inputs_preds_labels.jsonl"
        output_file_path = os.path.join(output_dir, output_file_path)
        with open(output_file_path, "w") as f_json:
            assert (
                len(outputs['inputs']) == len(outputs['preds']) == len(outputs['labels']) == len(outputs['metadata'])
            )
            for i, p, l, m in zip(outputs['inputs'], outputs['preds'], outputs['labels'], outputs['metadata']):
                json_string = {'input': i, 'pred_text': p, 'text': l}
                for k, v in m.items():
                    if k not in json_string:
                        json_string[k] = v
                f_json.write(json.dumps(json_string) + '\n')

        logging.info(f'Predictions saved to {output_file_path}')

    def setup_eval_dataloader(self, datasets, data_cfg):
        dataloaders = []
        if not isinstance(datasets, list):
            return self.build_data_loader(dataset=datasets, data_cfg=data_cfg, consumed_samples=0, is_eval=True)
        for dataset in datasets:
            eval_dl = self.build_data_loader(dataset=dataset, data_cfg=data_cfg, consumed_samples=0, is_eval=True)
            dataloaders.append(eval_dl)
        return dataloaders

    def get_test_dataloader(self, data_cfg):
        datasets = self._build_dataset(data_cfg, False)
        dataloaders = []
        if not isinstance(datasets, list):
            return self.build_data_loader(dataset=datasets, data_cfg=data_cfg, consumed_samples=0, is_eval=True)
        for dataset in datasets:
            eval_dl = self.build_data_loader(dataset=dataset, data_cfg=data_cfg, consumed_samples=0, is_eval=True)
            dataloaders.append(eval_dl)
        return dataloaders

    def sharded_state_dict(self, prefix: str = ''):
        """
        Force None for the parent class's sharded_state_dict() method if setup is complete.
        """
        if self.setup_complete:
            return None
        else:
            return super().sharded_state_dict(prefix=prefix)

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []

        model = PretrainedModelInfo(
            pretrained_model_name="speechllm_fc_llama2_7b",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:speechllm_fc_llama2_7b",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/speechllm_fc_llama2_7b/versions/1.24.0/files/speechllm_fc_llama2_7b.nemo",
        )
        results.append(model)
        return results


# TODO: https://www.diffchecker.com/b8kTpgGV/
class CrossAttendModularAudioGPTModel(ModularAudioGPTModel):
    """Modularized speech GPT model."""

    def prepare_llm_input(self, audio_batch):

        input_signal = audio_batch['audio_signal']
        input_signal_length = audio_batch['audio_signal_length']

        input_ids, input_length, labels, loss_mask = (
            audio_batch['tokens'],
            audio_batch['tokens_length'],
            audio_batch['labels'],
            audio_batch['loss_mask'],
        )

        num_audios = audio_batch.get("num_audios", None)
        if num_audios is not None:
            raise ValueError("num_audios is not supported.")

        if self.cfg.get('megatron_amp_O2', False):
            base_module = self.model.module
        else:
            base_module = self.model
        lm_embedding = (
            base_module.language_model.embedding if hasattr(base_module, 'language_model') else base_module.embedding
        )
        # [b, t, c]
        encoded, encoded_len = self.perception(
            input_signal=input_signal,
            input_signal_length=input_signal_length,
            processed_signal=None,
            processed_signal_length=None,
        )
        input_embeds = self._get_text_embeddings(input_ids, None).transpose(0, 1)
        encoder_input, extra_outputs = self.perception_cross_attn(encoded, encoded_len, input_embeds, input_lengths=input_length, return_mems=True)
        if 'audio_ratio' in audio_batch:
            audio_ratio = audio_batch['audio_ratio'][..., None, None]
            encoder_input = encoder_input * audio_ratio + input_embeds * (1 - audio_ratio)
        if 'alpha_xattn' in extra_outputs:
            alpha_xattn = extra_outputs['alpha_xattn']
            self.log(
                'alpha_xattn',
                alpha_xattn.mean(),
                prog_bar=True,
                batch_size=1,
                rank_zero_only=True,
            )
        attention_mask = self._create_attention_mask(encoder_input)

        if not hasattr(lm_embedding, 'transpose_batch_sequence') or lm_embedding.transpose_batch_sequence:
            encoder_input = encoder_input.transpose(0, 1).contiguous()
        if self.cfg.get("sequence_parallel", False):
            encoder_input = tensor_parallel.mappings.scatter_to_sequence_parallel_region(encoder_input)
        return encoder_input, attention_mask, labels, loss_mask, (encoded, encoded_len, extra_outputs)

    def setup_perception_modules(self, cfg):
        super().setup_perception_modules(cfg)
        imported_cls = model_utils.import_class_by_path(cfg.perception.xattn.target)
        self.perception_cross_attn = imported_cls(cfg=cfg.perception)

    def state_dict(self, destination=None, prefix=None, keep_vars=False):
        if self.setup_complete:
            return_state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
            state_dict = self.perception_cross_attn.state_dict(prefix="perception_cross_attn.")
            return_state_dict.update(state_dict)
            return return_state_dict
        else:
            return super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
