shots=( 16 32 48 64 80 96 112 )
for s in "${shots[@]}"
do
    CUDA_VISIBLE_DEVICES=3 \
    python entail2/runner/runner.py \
    --learning_rate 3e-5 \
    --warmup_ratio 0.006 \
    --train_batch_size 32 \
    --num_train_epochs 10 \
    --bert_name bart \
    --model_name unifew \
    --training_shots $s \
    --mode train;
    CUDA_VISIBLE_DEVICES=3 \
    python entail2/runner/runner.py \
    --learning_rate 3e-5 \
    --warmup_ratio 0.005 \
    --train_batch_size 32 \
    --num_train_epochs 10 \
    --bert_name bart \
    --model_name crossfit \
    --training_shots $s \
    --mode train;
done
# bash scripts/eval_models_on_tasks_0shot.sh