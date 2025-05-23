# python scripts/nlp_language_modeling/convert_hf_llama_to_nemo.py --in-file=/workspace/nemo/works/mod_speech_llm/models/llm/llm/tiny_llama/ --out-file=/workspace/nemo/works/mod_speech_llm/models/llm/llm/tiny_llama.nemo
NEMO_DIR=/workspace/nemo/works/zhehuaic_works/mod_speech_llm/NeMo_merge/
export PYTHONPATH=$NEMO_DIR:$PYTHONPATH

MEGATRON_CKPT=/media/data3/pretrained_models/megatron_gpt/gpt_pretrain_220m_len_4096_pos_alibi_step_595508_gbs256.nemo
MEGATRON_CKPT=../../../../models/llm/llama-2-7b.nemo
MEGATRON_CKPT=../../../../models/llm/llama-2-7b-chat.nemo
MEGATRON_CKPT=/workspace/nemo/works/mod_speech_llm/models/llm/llm/tiny_llama.nemo
#MEGATRON_CKPT=../../models/llm/megatron_gpt_sft_kickass_snake--validation_loss-1.886-step-1000-consumed_samples-127872.0.nemo
ASR_MODEL="ssl_en_conformer_large"
ASR_MODEL="../../../../models/llm/stt_en_fastconformer_transducer_large.nemo"
ASR_MODEL="stt_en_fastconformer_ctc_large"
ASR_MODEL="stt_en_fastconformer_transducer_large"
ASR_MODEL=/lustre/fs8/portfolios/llmservice/users/kpuvvada/results/canary-v0_XL_mLang_ASR-AST/oci_b6s4kf-ASR-AST_20240104_lfbe-128_ngpus-128_mbs-240s_opt-adamw_lr-3e-4_wd-1e-3_sched-InverseSquareRootAnnealing_maxsteps-100000/oci_b6s4kf-ASR-AST_20240104_lfbe-128_ngpus-128_mbs-240s_opt-adamw_lr-3e-4_wd-1e-3_sched-InverseSquareRootAnnealing_maxsteps-100000/checkpoints/oci_b6s4kf-ASR-AST_20240104_lfbe-128_ngpus-128_mbs-240s_opt-adamw_lr-3e-4_wd-1e-3_sched-InverseSquareRootAnnealing_maxsteps-100000-averaged.nemo
ASR_MODEL=/workspace/nemo/works/zhehuaic_works/llm/oci_b6s4kf-ASR-AST_20240104_lfbe-128_ngpus-128_mbs-240s_opt-adamw_lr-3e-4_wd-1e-3_sched-InverseSquareRootAnnealing_maxsteps-100000-averaged.nemo
ASR_MODEL=/workspace/nemo/works/zhehuaic_works/llm/canary-1b.nemo

GLOBAL_BATCH=2
MICRO_BATCH=2

TRAIN_MANIFESTS=/media/data/datasets/LibriSpeech/dev_clean.json
TRAIN_MANIFESTS=/media/data/datasets/LibriSpeech/dev_clean_150_r.json
TRAIN_MANIFESTS=/media/data/datasets/LibriSpeech/dev_clean_300.json
TRAIN_MANIFESTS=/media/data/datasets/LibriSpeech/train_clean_100_cleaned.json
TRAIN_MANIFESTS=/media/data/datasets/LibriSpeech/dev_clean_140_r.shuf.json
TRAIN_MANIFESTS=/media/data/datasets/LibriSpeech/dev_clean_10.json
TRAIN_MANIFESTS=[/media/data/datasets/LibriSpeech/dev_clean_10.json,/media/data/datasets/LibriSpeech/dev_clean_11.json]
TRAIN_MANIFESTS=[/media/data/datasets/LibriSpeech/dev_clean_10_text.json,/media/data/datasets/LibriSpeech/dev_clean_10_text.json]
train_questions=[/media/data/datasets/LibriSpeech/dev_clean_10_q_set.json,/media/data/datasets/LibriSpeech/dev_clean_11_q_set.json]
TRAIN_MANIFESTS=[/media/data/datasets/LibriSpeech/dev_clean_10_text.json,/media/data/datasets/LibriSpeech/dev_clean_10.json]
TRAIN_MANIFESTS=[[manifests/covost_v2.es_en.dev.es.40.json,1],[manifests/covost_v2.es_en.dev.es.40.json,1]]
TRAIN_MANIFESTS=[[/media/data/datasets/LibriSpeech/dev_clean_10.json,1],[/media/data/datasets/LibriSpeech/dev_clean_10.json,1]]
VAL_MANIFESTS=/media/data/datasets/LibriSpeech/dev_clean_150_r.json
VAL_MANIFESTS=/media/data/datasets/LibriSpeech/dev_clean_2.json
VAL_MANIFESTS=/media/data/datasets/LibriSpeech/dev_clean_300.json
VAL_MANIFESTS=/media/data/datasets/LibriSpeech/dev_clean.json

VAL_MANIFESTS=/media/data/datasets/LibriSpeech/dev_clean_10.json
VAL_MANIFESTS=[/media/data/datasets/LibriSpeech/dev_clean_10.json,/media/data/datasets/LibriSpeech/dev_clean_11.json]
valid_questions=[/media/data/datasets/LibriSpeech/dev_clean_10_q_set.json,/media/data/datasets/LibriSpeech/dev_clean_11_q_set.json]
VAL_MANIFESTS=[/media/data/datasets/LibriSpeech/dev_clean_10_text.json,/media/data/datasets/LibriSpeech/dev_clean_10.json]
VAL_MANIFESTS=[manifests/covost_v2.es_en.dev.es.40.json]
VAL_MANIFESTS=[/media/data/datasets/LibriSpeech/dev_clean_10.json,/media/data/datasets/LibriSpeech/dev_clean_10.json]

#python \

export NVTE_MASKED_SOFTMAX_FUSION=0
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=0

#	python \
python -m pdb -c continue \
../speech_llm/modular_audio_gpt_train.py --config-path="../speech_llm/conf/" --config-name "modular_audio_gpt_config_cross_llama_lhotse" \
    model.perception.xattn.target=nemo.collections.multimodal.speech_llm.modules.perception.ProjectTransformerCrossAttention \
    ++model.perception.xattn.xformer_num_layers=1 \
    ++model.data.train_ds.num_workers=0 \
    model.freeze_llm=false \
    model.megatron_amp_O2=false \
    model.optim.name=distributed_fused_adam \
        ++model.optim.bucket_cap_mb=200 \
    ++model.optim.overlap_grad_sync=False \
    ++model.optim.contiguous_grad_buffer=True \
    ++model.use_flash_attention=True \
    model.freeze_audio_encoder=false \
    model.pretrained_audio_model=$ASR_MODEL \
    model.restore_from_path=$MEGATRON_CKPT \
    model.global_batch_size=$GLOBAL_BATCH \
    model.micro_batch_size=$MICRO_BATCH \
    ++trainer.use_distributed_sampler=false \
    ++trainer.limit_train_batches=10 \
    ++trainer.val_check_interval=10 \
    ~model.peft \
    model.perception.modality_adapter.subsampling_factor=8 \
    model.perception.modality_adapter.reduction_factor=8 \
    model.perception.modality_adapter.reduction=striding \
    model.perception.modality_adapter.reduction_position=-1 \
    ++model.perception.use_multi_layer_feat=false \
    ++model.perception.add_sep=true \
    ++model.perception.multi_layer_feat.layer_idx_list='[0,6,12,16,-1]' \
    ++model.perception.multi_layer_feat.aggregator.align_mode=max \
    ++model.perception.is_canary=True \
    ++model.perception.is_ctc=False \
    ++model.perception.greedy_decoding_overwrite=true \
    ++model.data.train_ds.convert_canary_prompt_to_text=true \
    ++model.data.validation_ds.convert_canary_prompt_to_text=true \
    ++model.data.train_ds.seed='trng' \
    ++model.data.train_ds.use_bucketing=false  \
    ++model.data.train_ds.canary_tokens_augment_ratio=0.5 \
    ++model.data.train_ds.batch_size=2 \
    ++model.data.train_ds.batch_duration=null \
    ++model.data.validation_ds.max_seq_length=512 \
    ++model.data.train_ds.add_bos=True \
    ++model.data.validation_ds.random_context_prob=0.5 \
    ++model.data.validation_ds.drop_last=False \
    ++model.data.validation_ds.use_lhotse=True \
    ++model.data.validation_ds.batch_size=$MICRO_BATCH \
    ++model.data.validation_ds.use_bucketing=False \
    ++model.data.validation_ds.text_field=text \
    ++model.data.train_ds.text_field=text \
    model.data.train_ds.manifest_filepath=$TRAIN_MANIFESTS \
    model.data.validation_ds.manifest_filepath=$VAL_MANIFESTS

    #++model.data.train_ds.prepend_to_exist_question='Transcribe the spoken content to written English text ignoring punctuations and capitalization and then answer the following question: ' \
    #++model.data.validation_ds.prepend_to_exist_question='Transcribe the spoken content to written English text ignoring punctuations and capitalization and then answer the following question: ' \

