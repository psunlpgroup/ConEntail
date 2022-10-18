# cd ..

# TASKS=(glue-cola glue-qqp glue-sst2 glue-mrpc scitail amazon_polarity)
# TASKS=(tweet_eval-irony ag_news rotten_tomatoes hate_speech_offensive imdb)
# TASKS=(ag_news rotten_tomatoes hate_speech_offensive imdb)
# TASKS=(ag_news)
# TASKS=(glue-cola amazon_polarity rotten_tomatoes)
# TASKS=(glue-sst2 glue-mrpc scitail ag_news hate_speech_offensive glue-qqp)
# TASKS=( glue-sst2 glue-mrpc scitail ag_news hate_speech_offensive )
# TASKS=( glue-sst2 glue-mrpc scitail )
TASKS=( ag_news hate_speech_offensive )
# SHOTS=(5 10 20 30 40 50 60 70 80 90 100 110)
# SHOTS=(90 100 110)
# SHOTS=(5 20 30 40 50 60 70 80 90)
# SHOTS=( 40 50 60 70 80 )
SHOTS=( 80 70 60 )

# SHOTS=(100 110)
# TASKS=(trec-finegrained)
# MODELS=(entail)
MODELS=(unifew crossfit)

# MODELS=(entail2 efl crossfit unifew)
# MODELS=(unifew crossfit)
# MODELS=(efl entail2)
# MODELS=(efl)

GPU=3

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
            --learning_rate 3e-5 \
            --warmup_ratio 0.06 \
            --num_train_epochs 400 \
            --test_times 3 \
            --test_shots ${SHOT} \
            --mode finetune_test
        done
    done
done


# RuntimeError: CUDA out of memory. Tried to allocate 5.75 GiB (GPU 0; 47.54 GiB total capacity; 44.47 GiB already allocated; 829.38 MiB free; 44.86 GiB reserved in total by PyTorch)
