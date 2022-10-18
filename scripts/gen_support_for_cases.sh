TASKS=(ade_corpus_v2-classification ag_news amazon_polarity anli circa climate_fever dbpedia_14 discovery emo emotion ethos-directed_vs_generalized ethos-disability ethos-gender ethos-national_origin ethos-race ethos-religion ethos-sexual_orientation financial_phrasebank glue-cola glue-mnli glue-mrpc glue-qnli glue-qqp glue-rte glue-sst2 glue-wnli google_wellformed_query hate_speech18 hate_speech_offensive hatexplain imdb kilt_fever liar medical_questions_pairs onestop_english paws poem_sentiment rotten_tomatoes scicite scitail sick sms_spam superglue-cb superglue-rte superglue-wic superglue-wsc tab_fact trec trec-finegrained tweet_eval-emoji tweet_eval-emotion tweet_eval-hate tweet_eval-irony tweet_eval-offensive tweet_eval-sentiment tweet_eval-stance_abortion tweet_eval-stance_atheism tweet_eval-stance_climate tweet_eval-stance_feminist tweet_eval-stance_hillary wiki_qa yahoo_answers_topics yelp_polarity)
for TASK in ${TASKS[@]};
do
    python scripts/gen_singletask_zeroshot_support.py \
    --data_dir raw_data/gym \
    --task_dir ${TASK} --shots 1 --times 1
done