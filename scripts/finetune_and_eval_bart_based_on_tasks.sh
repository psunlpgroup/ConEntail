# cd ..

# TASKS=(trec trec-finegrained glue-cola glue-qqp glue-sst2 glue-mrpc scitail amazon_polarity)
TASKS=(glue-cola glue-qqp glue-sst2 glue-mrpc scitail amazon_polarity ag_news rotten_tomatoes hate_speech_offensive imdb)
SHOT=10
# MODEL=entail2
# MODELS=(entail2 efl crossfit unifew)
MODELS=(unifew crossfit)
# MODELS=(efl)

GPU=2
for TASK in ${TASKS[@]};
do
    # python scripts/gen_singletask_test.py \
    # --data_dir raw_data/gym \
    # --task_dir ${TASK}
    # python scripts/gen_singletask_support.py \
    # --data_dir raw_data/gym \
    # --task_dir ${TASK} --shots ${SHOT} --times 3    
    for MODEL in ${MODELS[@]};
    do
        echo "Task: $TASK, Model: $MODEL"
        CUDA_VISIBLE_DEVICES=$GPU \
        python entail2/runner/runner.py \
        --data_dir raw_data/gym \
        --task_dir ${TASK} \
        --model ${MODEL} \
        --learning_rate 3e-5 \
        --warmup_ratio 0.06 \
        --num_train_epochs 400 \
        --test_times 10 \
        --test_shots ${SHOT} \
        --mode finetune_test
    done
done
# for TASK in ${TASKS[@]};
# do
#     echo "Task: $TASK, Checkpoint: $CHECKPOINT, Identifier: $IDENTIFIER"
# done