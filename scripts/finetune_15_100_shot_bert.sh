# cd ..

# TASKS=(glue-cola glue-qqp glue-sst2 glue-mrpc scitail amazon_polarity)
# TASKS=(tweet_eval-irony ag_news rotten_tomatoes hate_speech_offensive imdb)
# TASKS=(ag_news rotten_tomatoes hate_speech_offensive imdb)
# TASKS=(ag_news)
# TASKS=(glue-cola amazon_polarity rotten_tomatoes)
# TASKS=(rotten_tomatoes)
# TASKS=(glue-sst2 glue-mrpc scitail ag_news hate_speech_offensive glue-qqp)
# TASKS=(glue-sst2 glue-mrpc scitail ag_news hate_speech_offensive glue-qqp)
# TASKS=( glue-sst2 glue-mrpc scitail )
TASKS=( ag_news hate_speech_offensive glue-qqp )
# SHOTS=(5 20 30 40 50 60 70 80 90 100 110)
# SHOTS=( 5 20 30 40 50 60 70 80 90)
# SHOTS=( 30 40 )
# SHOTS=( 70 80 )
SHOTS=( 80 70 60 )

# SHOTS=(50 60 70)

# SHOTS=(100 110)
# TASKS=(trec-finegrained)
# MODELS=(entail)
# MODELS=(unifew crossfit)

# MODELS=(entail2 efl crossfit unifew)
# MODELS=(unifew crossfit)
# MODELS=(efl entail2)
# MODELS=(efl_no_cl entail2)
MODELS=( efl_no_cl entail2 )
# MODELS=(entail2)

GPU=2

for SHOT in ${SHOTS[@]};
do
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
            --learning_rate 1e-5 \
            --warmup_ratio 0.06 \
            --num_train_epochs 10 \
            --test_times 3 \
            --test_shots ${SHOT} \
            --mode finetune_test
        done
    done
done