# shots=( 16 32 48 64 80 96 112 )
shots=( 64 )
for s in "${shots[@]}"
do
    CUDA_VISIBLE_DEVICES=1 \
    python entail2/runner/runner.py \
    --learning_rate 1e-5 \
    --warmup_ratio 0.06 \
    --train_batch_size 32 \
    --num_train_epochs 10 \
    --bert_name bert \
    --model_name efl_no_cl \
    --training_shots $s \
    --mode train;
    CUDA_VISIBLE_DEVICES=1 \
    # python entail2/runner/runner.py \
    # --learning_rate 1e-5 \
    # --warmup_ratio 0.06 \
    # --train_batch_size 32 \
    # --num_train_epochs 10 \
    # --bert_name bert \
    # --model_name entail2 \
    # --use_sampler \
    # --training_shots $s \
    # --mode train;
done
# bash scripts/eval_models_on_tasks_0shot.sh