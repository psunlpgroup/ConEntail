GPU=1
# CUDA_VISIBLE_DEVICES=$GPU \
# python entail2/runner/runner.py \
# --model entail2 \
# --test_times 1 \
# --test_shots 1 \
# --training_shots 128 \
# --case_support_file sentiment.json \
# --case_test_file test.json \
# --mode case

CUDA_VISIBLE_DEVICES=$GPU \
python entail2/runner/runner.py \
--model entail2 \
--test_times 1 \
--test_shots 1 \
--training_shots 128 \
--case_support_file topic_support.json \
--case_test_file topic_test.json \
--mode case

# CUDA_VISIBLE_DEVICES=$GPU \
# python entail2/runner/runner.py \
# --model entail2 \
# --test_times 1 \
# --test_shots 1 \
# --training_shots 128 \
# --case_support_file others.json \
# --case_test_file test.json \
# --mode case

# CUDA_VISIBLE_DEVICES=$GPU \
# python entail2/runner/runner.py \
# --model entail2 \
# --test_times 1 \
# --test_shots 1 \
# --training_shots 128 \
# --case_support_file covid_support.json \
# --case_test_file covid_test.json \
# --mode case