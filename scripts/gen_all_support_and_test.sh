TASKS=(glue-cola glue-qqp glue-sst2 glue-mrpc scitail amazon_polarity ag_news rotten_tomatoes hate_speech_offensive)
for TASK in ${TASKS[@]};
do
    python scripts/gen_singletask_test.py \
    --data_dir raw_data/gym \
    --task_dir ${TASK}
    python scripts/gen_singletask_support.py \
    --data_dir raw_data/gym \
    --task_dir ${TASK} --shots 1 --times 10 
    python scripts/gen_singletask_support.py \
    --data_dir raw_data/gym \
    --task_dir ${TASK} --shots 5 --times 10  
    python scripts/gen_singletask_support.py \
    --data_dir raw_data/gym \
    --task_dir ${TASK} --shots 10 --times 10   
done