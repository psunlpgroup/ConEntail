CUDA_VISIBLE_DEVICES=3 \
python entail2/runner/runner.py \
--learning_rate 1e-5 \
--warmup_ratio 0.06 \
--train_batch_size 32 \
--num_train_epochs 10 \
--bert_name bert \
--model_name efl_no_cl \
--mode train;
# bash scripts/eval_models_on_tasks_0shot.sh