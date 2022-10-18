# cd ..

# TASKS=(glue-cola glue-qqp glue-sst2 glue-mrpc scitail amazon_polarity)
# TASKS=(tweet_eval-irony ag_news rotten_tomatoes hate_speech_offensive imdb)
# TASKS=(ag_news rotten_tomatoes hate_speech_offensive imdb)
# TASKS=(ag_news)
TASKS=(glue-cola glue-qqp glue-sst2 glue-mrpc scitail amazon_polarity ag_news rotten_tomatoes hate_speech_offensive)
# TASKS=(glue-cola)
SHOT=10

# TASKS=(trec-finegrained)
# MODELS=(entail2 efl_multichoice efl)
# MODELS=(entail2 efl crossfit unifew)
# MODELS=(unifew crossfit)
# MODELS=(efl entail2)
MODELS=(efl_no_cl)
# MODELS=(efl)
# MODELS=(entail2)

GPU=2

for TASK in ${TASKS[@]};
do
    # python scripts/gen_singletask_test.py \
    # --data_dir raw_data/gym \
    # --task_dir ${TASK}
    python scripts/gen_singletask_support.py \
    --data_dir raw_data/gym \
    --task_dir ${TASK} --shots ${SHOT} --times 10    
    for MODEL in ${MODELS[@]};
    do
        echo "Task: $TASK, Model: $MODEL"
        CUDA_VISIBLE_DEVICES=$GPU \
        python entail2/runner/runner.py \
        --data_dir raw_data/gym \
        --task_dir ${TASK} \
        --model ${MODEL} \
        --learning_rate 1e-5 \
        --warmup_ratio 0.06 \
        --num_train_epochs 10 \
        --test_times 10 \
        --test_shots ${SHOT} \
        --mode finetune_test
    done
done
