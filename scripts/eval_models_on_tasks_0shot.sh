# cd ..

# TASKS=(trec trec-finegrained glue-cola glue-qqp glue-sst2 glue-mrpc scitail amazon_polarity rotten_tomatoes hate_speech_offensive imdb)
# TASKS=(ag_news)
# TASKS=(glue-cola amazon_polarity rotten_tomatoes glue-mrpc scitail ag_news hate_speech_offensive)
# TASKS=(glue-qqp glue-sst2)
TASKS=(hate_speech_offensive)
# TASKS=(glue-mrpc glue-qqp amazon_polarity hate_speech_offensive)
# TASKS=(glue-cola glue-qqp glue-sst2 glue-mrpc scitail amazon_polarity ag_news rotten_tomatoes hate_speech_offensive imdb)
MODELS=(entail2)
# MODELS=(efl_no_cl entail2 crossfit unifew)
# MODELS=(unifew crossfit)
# MODELS=(unifew)
# MODELS=(efl_no_cl)
# MODELS=(efl)

IDENTIFIER=3
GPU=0
for TASK in ${TASKS[@]};
do
    python scripts/gen_singletask_test.py \
    --data_dir raw_data/gym \
    --task_dir ${TASK}
    python scripts/gen_singletask_zeroshot_support.py \
    --data_dir raw_data/gym \
    --task_dir ${TASK} --shots 1 --times 1
    for MODEL in ${MODELS[@]};
    do
        echo "Task: $TASK, Model: $MODEL"

        CUDA_VISIBLE_DEVICES=$GPU \
        python entail2/runner/runner.py \
        --data_dir raw_data/gym \
        --task_dir ${TASK} \
        --model ${MODEL} \
        --test_times 1 \
        --test_shots 1 \
        --mode test
    done
done
# for TASK in ${TASKS[@]};
# do
#     echo "Task: $TASK, Checkpoint: $CHECKPOINT, Identifier: $IDENTIFIER"
# done