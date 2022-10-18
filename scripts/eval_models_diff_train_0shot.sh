# cd ..

# TASKS=(glue-cola amazon_polarity rotten_tomatoes glue-mrpc scitail glue-sst2)
# TASKS=(rotten_tomatoes glue-mrpc scitail glue-sst2)
TASKS=(ag_news hate_speech_offensive glue-qqp)
# TASKS=(hate_speech_offensive)
# TASKS=(glue-mrpc glue-qqp amazon_polarity hate_speech_offensive)
# TASKS=(glue-cola glue-qqp glue-sst2 glue-mrpc scitail amazon_polarity ag_news rotten_tomatoes hate_speech_offensive imdb)
# MODEL=entail2
# MODELS=(entail2 efl_no_cl crossfit unifew)
MODELS=( entail2 efl_no_cl )
# MODELS=(unifew crossfit)
# MODELS=(unifew)
# MODELS=(efl_multichoice)
# MODELS=(efl_no_cl)
# MODELS=(efl)
shots=( 32 48 64 80 96 112 )
# shots=( 64 )

IDENTIFIER=3
GPU=1
for MODEL in ${MODELS[@]};
do
    python scripts/gen_singletask_test.py \
    --data_dir raw_data/gym \
    --task_dir ${TASK}
    python scripts/gen_singletask_zeroshot_support.py \
    --data_dir raw_data/gym \
    --task_dir ${TASK} --shots 1 --times 1
    for TASK in ${TASKS[@]};
    do
        for s in "${shots[@]}";
        do
            echo "Task: $TASK, Model: $MODEL, Training_shots: $s"
            CUDA_VISIBLE_DEVICES=$GPU \
            python entail2/runner/runner.py \
            --data_dir raw_data/gym \
            --task_dir ${TASK} \
            --model ${MODEL} \
            --test_times 1 \
            --test_shots 1 \
            --training_shots $s \
            --mode test
        done
    done
done
# for TASK in ${TASKS[@]};
# do
#     echo "Task: $TASK, Checkpoint: $CHECKPOINT, Identifier: $IDENTIFIER"
# done