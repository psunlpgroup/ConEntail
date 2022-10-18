CUDA_VISIBLE_DEVICES=2 \
python entail2/runner/runner.py \
--data_dir raw_data/gym \
--task_dir trec \
--learning_rate 1e-5 \
--warmup_ratio 0.06 \
--num_train_epochs 10 \
--model_name entail2 \
--test_times 3 \
--test_shots 1 \
--mode finetune_test;