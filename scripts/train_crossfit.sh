CUDA_VISIBLE_DEVICES=2 \
python entail2/runner/runner.py \
--learning_rate 3e-5 \
--warmup_ratio 0.005 \
--train_batch_size 32 \
--num_train_epochs 10 \
--bert_name bart \
--model_name crossfit \
--mode train;
