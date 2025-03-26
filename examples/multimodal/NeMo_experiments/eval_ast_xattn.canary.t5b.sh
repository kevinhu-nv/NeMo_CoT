NEMO_DIR=/workspace/nemo/works/zhehuaic_works/mod_speech_llm/NeMo_merge
export PYTHONPATH=$NEMO_DIR:$PYTHONPATH
echo $PYTHONPATH

MEGATRON_CKPT=/workspace/nemo/works/mod_speech_llm/models/llm/llm/alma7b.nemo
MEGATRON_CKPT=/workspace/nemo/works/mod_speech_llm/models/llm/llm/tiny_llama.nemo
MEGATRON_CKPT=/workspace/nemo/works/mod_speech_llm/models/llm/llm/megatron_nmt_any_any.mengru.nemo
#alm_path=nemo_experiments/megatron_audio_gpt_peft_tuning/
#ALM_CKPT=$alm_path"/checkpoints/megatron_audio_gpt_peft_tuning--validation_bleu\=49.967-step\=1450-epoch\=144-last.ckpt"
#ALM_YAML=$alm_path'/version_0/hparams.yaml'
name=crossbmg4eghel_t5jae_lhmerge_oci_FC-GPT_t5_x_x_canaryset_b6s4kf-sunolong_noCC_langtemp0.5_dsettemp0.5_lr1e-4wd0_CosineAnnealing_warmup2500_minlr1e-6_gbs2048_mbs16_ep200
cluster=oci
if [ $cluster = oci ]; then
ALM_CKPT=`ls -lrt /lustre/fs8/portfolios/llmservice/users/zhehuaic/results/canary-v0_speechllm/$name/$name/checkpoints/* | grep -v last | tail -n 1 | awk '{print $NF}' | sed 's/=/\\\\=/g'`
else
ALM_CKPT=`ls -lrt /gpfs/fs1/projects/ent_aiapps/users/zhehuaic/results/audio-text-llm-debug/$name/$name/checkpoints/* | grep -v last | tail -n 1 | awk '{print $NF}' | sed 's/=/\\\\=/g'`
fi
ALM_YAML=`dirname $ALM_CKPT`/../version_1/hparams.yaml
ALM_CKPT=/workspace/nemo/works/zhehuaic_works/llm/crossbmg4eghel_t5jae_lhmerge_oci_FC-GPT_t5_x_x_canaryset_b6s4kf-sunolong_noCC_langtemp0.5_dsettemp0.5_lr1e-4wd0_CosineAnnealing_warmup2500_minlr1e-6_gbs2048_mbs16_ep200/megatron_audio_gpt_peft_tuning--validation_bleu\\=65.003-step\\=40004-epoch\\=1.ckpt

# local overwrite
#ALM_CKPT=/media/zhehuaic_works/llm/crossbmg4kc3d_lhmain3cross_oci_FC-GPT_alma7b_canaryset_b6s4kf-ASR-AST_lr5e-4wd1e-5_CosineAnnealing_warmup2000_minlr1e-4_gbs256_mbs8_ep200/megatron_audio_gpt_peft_tuning--val_loss\\=0.898-step\\=6001-epoch\\=2.ckpt


ASR_MODEL=/workspace/nemo/works/zhehuaic_works/llm/canary-1b.nemo
#ASR_MODEL=stt_en_fastconformer_transducer_large

# VAL_MANIFESTS="[/media/data/datasets/LibriSpeech/debug_1.json,/media/data/datasets/LibriSpeech/debug_1.json]"
# VAL_NAMES=[debug-1,debug-1]

# VAL_MANIFESTS="[/media/data/datasets/LibriSpeech/dev_clean_cleaned.json,/media/data/datasets/LibriSpeech/dev_other.json,/media/data/datasets/LibriSpeech/train_clean_100_cleaned.json]"
# VAL_NAMES="[dev-clean,dev-other,train-clean-100]"

VAL_MANIFESTS="[/media/data/datasets/LibriSpeech/dev_clean_10_q.json]"
VAL_NAMES="[dev_clean_10]"
VAL_MANIFESTS="[/media/data/datasets/LibriSpeech/test_clean_32.json]"
VAL_NAMES="[test_clean_32]"

VAL_MANIFESTS="[/media/data/datasets/LibriSpeech/test_clean_q.json,/media/data/datasets/LibriSpeech/test_clean_q.json]"
VAL_NAMES="[test_clean_q,text.1a.json]"
valid_questions=[/media/data/datasets/LibriSpeech/dev_clean_10_q_set.json,/media/data/datasets/LibriSpeech/dev_clean_11_q_set.json]
VAL_MANIFESTS=[/media/data/datasets/LibriSpeech/dev_clean_2.json,manifests/text.1a.json]
VAL_MANIFESTS=[/media/data/datasets/LibriSpeech/dev_clean_10.json,/media/data/datasets/LibriSpeech/dev_clean_10.json]
VAL_MANIFESTS=[manifests/ast.1b.json,manifests/ast.1c.json,manifests/ast.1d.json]
VAL_NAMES=[en_zh0,en_ja0,de_zh0]

field_name=text_trans
VAL_MANIFESTS=[manifests/ast.dezh.1b.sh,manifests/ast.dezh.1c.sh,manifests/ast.dezh.1d.sh]
VAL_NAMES=[de_zh0,de_zh,de_zhd]

field_name=text
VAL_NAMES=[SPGI_testall]
VAL_MANIFESTS=[/workspace/nemo/works/zhehuaic_works/data/SPGI.json]
VAL_NAMES=[en_de,en_zh0,en_zh,en_ja0,en_ja,en_jad]
VAL_MANIFESTS=[manifests/ast.1a2.json,manifests/ast.1b.json,manifests/ast.1c.json,manifests/ast.ja.1b.json,manifests/ast.ja.1c.json,manifests/ast.ja.1d.json]
VAL_NAMES=[en_de,en_de2]
VAL_MANIFESTS=[manifests/ast.1a2.json,manifests/ast.1a.json]
VAL_NAMES=[en_de_all_tst2]
VAL_MANIFESTS=[/dev_data/st_dev/tst-COMMON_manifest.json]
VAL_Q=[manifests/ast.1a2.q]
VAL_NAMES=[en_de_all_tst_asr]
VAL_MANIFESTS=[/dev_data/st_dev/tst-COMMON_manifest.json]
VAL_Q=[manifests/asr.q]
#VAL_NAMES=[all_positive_test.noquestion]
#VAL_MANIFESTS=[/gpfs/fs1/projects/ent_aiapps/users/zhehuaic/works/mod_speech_llm/data/msmarco_train_not_normalized/all_positive_test.noquestion.json]
#VAL_Q=[manifests/asr.q]

VAL_MANIFESTS=['/home_1/aandrusenko/tools/nemo_recipes/librispeech/data/gtc_combined/combined_gtc_manifest.json.filt_ml-20.json']
VAL_NAMES=[gtc1a]
VAL_Q=[/questions/asr.en.questions_gtc1a]
VAL_MANIFESTS=['/home_1/aandrusenko/tools/nemo_recipes/librispeech/data/gtc_combined/combined_gtc_manifest.json.filt_ml-20.json','/home_1/aandrusenko/tools/nemo_recipes/librispeech/data/gtc_combined/combined_gtc_manifest.json.filt_ml-20.json','/home_1/aandrusenko/tools/nemo_recipes/librispeech/data/gtc_combined/combined_gtc_manifest.json.filt_ml-20.json']
VAL_NAMES=[gtc1e,gtc1c,gtc1a]
VAL_Q=[/questions/asr.en.questions_gtc1e_c,/questions/asr.en.questions_gtc1c_c,/questions/asr.en.questions_gtc1a_c]
VAL_MANIFESTS=['/workspace/nemo/works/zhehuaic_works/data/conec_kaldi_data_small_salm/conec_small_gt.slides.c.txt','/workspace/nemo/works/zhehuaic_works/data/conec_kaldi_data_small_salm/conec_small_slides.c.txt']
VAL_NAMES=[conec_small_gt.slides.txt,conec_small_slides.txt]
VAL_MANIFESTS=['/workspace/nemo/works/zhehuaic_works/data/conec_kaldi_data_small_salm/conec_small_nb.c.txt','/workspace/nemo/works/zhehuaic_works/data/conec_kaldi_data_small_salm/conec_small_gt.c.txt','/workspace/nemo/works/zhehuaic_works/data/conec_kaldi_data_small_salm/conec_small_gt.slides.c.txt']
VAL_NAMES=[conec_small_nb.txt,conec_small_gt.txt,conec_small_gt.slides.txt]
VAL_MANIFESTS=[/lustre/fs7/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/AST/fleurs/manifests/fleurs.en_fr.test.json,/lustre/fs7/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/AST/fleurs/manifests/fleurs.en_de.test.json,/lustre/fs7/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/AST/fleurs/manifests/fleurs.en_es.test.json,/lustre/fs7/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/AST/fleurs/manifests/fleurs.fr_en.test.json,/lustre/fs7/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/AST/fleurs/manifests/fleurs.de_en.test.json,/lustre/fs7/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/AST/fleurs/manifests/fleurs.es_en.test.json]
VAL_NAMES=[en_fr,en_de,en_es,fr_en,de_en,es_en]
field_name=answer

EXP_NAME=ast.`echo $ALM_YAML | awk -F '/' '{print $(NF-4)}'`
mkdir -p results/$EXP_NAME/

batch=8

set -x

pip install kaldialign

export NVTE_MASKED_SOFTMAX_FUSION=0
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=0

stage=1
if [ $stage -le 1 ] ; then

HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=1
#python -m pdb -c continue \
python \
	../speech_llm/modular_audio_gpt_eval.py \
    model.restore_from_path=$MEGATRON_CKPT \
    model.peft.restore_from_path="$ALM_CKPT" \
    model.peft.restore_from_hparams_path=$ALM_YAML \
    ++model.pretrained_audio_model=$ASR_MODEL \
    model.data.test_ds.manifest_filepath=$VAL_MANIFESTS \
    ++model.data.test_ds.deduplicate=False \
    model.data.test_ds.names=$VAL_NAMES \
    model.global_batch_size=$batch \
    model.micro_batch_size=$batch \
    ++model.data.test_ds.drop_last=False \
    model.data.test_ds.micro_batch_size=$batch \
	model.data.test_ds.global_batch_size=$batch \
    ++model.data.test_ds.lang_field="target_lang" \
    ++model.data.test_ds.text_field="answer" \
    ++model.data.test_ds.use_lhotse=True \
    ++model.data.test_ds.convert_canary_prompt_to_text=true \
    ++model.data.test_ds.end_string=null \
    ++model.data.test_ds.batch_size=$batch \
    ++model.data.test_ds.use_bucketing=False \
    ++model.data.test_ds.text_field=$field_name \
    ++inference.greedy=True \
    ++trainer.precision=bf16 \
  ++model.data.test_ds.metric.name='bleu' \
    ++inference.temperature=0.4 \
    ++inference.top_k=50 \
    ++inference.top_p=0.95 \
	model.data.test_ds.tokens_to_generate=128 \
  ++exp_manager.name=$EXP_NAME \
     model.data.test_ds.output_file_path_prefix=results/$EXP_NAME/ 2>&1 | tee results/$EXP_NAME/inf.log
echo $ALM_CKPT results/$EXP_NAME/inf.log
fi

exit

if [ $stage -le 2 ] ; then
KW_file=kwboost/boost_list.63
KW_file=kwboost/boost_list.31
for i in `ls results/$EXP_NAME/_test*jsonl`; do
     name=`basename $i`
 python kwboost/compute_key_words_fscore3.py --key_words_file $KW_file  --input_manifest $i > $i.out
     echo $name $i.out `cat $i.out| grep Fscore`
done
fi
