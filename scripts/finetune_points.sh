GPU=3
TASK=glue-qqp


CUDA_VISIBLE_DEVICES=$GPU \
python entail2/runner/runner.py \
--data_dir raw_data/gym \
--task_dir ${TASK} \
--model entail2 \
--learning_rate 1e-5 \
--warmup_ratio 0.06 \
--num_train_epochs 10 \
--test_times 10 \
--test_shots 30 \
--mode finetune_test

CUDA_VISIBLE_DEVICES=$GPU \
python entail2/runner/runner.py \
--data_dir raw_data/gym \
--task_dir ${TASK} \
--model entail2 \
--learning_rate 1e-5 \
--warmup_ratio 0.06 \
--num_train_epochs 10 \
--test_times 10 \
--test_shots 40 \
--mode finetune_test

CUDA_VISIBLE_DEVICES=$GPU \
python entail2/runner/runner.py \
--data_dir raw_data/gym \
--task_dir ${TASK} \
--model efl_no_cl \
--learning_rate 1e-5 \
--warmup_ratio 0.06 \
--num_train_epochs 10 \
--test_times 10 \
--test_shots 30 \
--mode finetune_test

CUDA_VISIBLE_DEVICES=$GPU \
python entail2/runner/runner.py \
--data_dir raw_data/gym \
--task_dir ${TASK} \
--model efl_no_cl \
--learning_rate 1e-5 \
--warmup_ratio 0.06 \
--num_train_epochs 10 \
--test_times 10 \
--test_shots 40 \
--mode finetune_test




CUDA_VISIBLE_DEVICES=$GPU \
python entail2/runner/runner.py \
--data_dir raw_data/gym \
--task_dir ${TASK} \
--model crossfit \
--learning_rate 3e-5 \
--warmup_ratio 0.06 \
--num_train_epochs 400 \
--test_times 3 \
--test_shots 20 \
--mode finetune_test

CUDA_VISIBLE_DEVICES=$GPU \
python entail2/runner/runner.py \
--data_dir raw_data/gym \
--task_dir ${TASK} \
--model crossfit \
--learning_rate 3e-5 \
--warmup_ratio 0.06 \
--num_train_epochs 400 \
--test_times 3 \
--test_shots 30 \
--mode finetune_test

CUDA_VISIBLE_DEVICES=$GPU \
python entail2/runner/runner.py \
--data_dir raw_data/gym \
--task_dir ${TASK} \
--model unifew \
--learning_rate 3e-5 \
--warmup_ratio 0.06 \
--num_train_epochs 400 \
--test_times 3 \
--test_shots 20 \
--mode finetune_test

CUDA_VISIBLE_DEVICES=$GPU \
python entail2/runner/runner.py \
--data_dir raw_data/gym \
--task_dir ${TASK} \
--model unifew \
--learning_rate 3e-5 \
--warmup_ratio 0.06 \
--num_train_epochs 400 \
--test_times 3 \
--test_shots 30 \
--mode finetune_test