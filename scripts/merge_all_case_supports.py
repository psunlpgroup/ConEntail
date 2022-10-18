import os

all_tasks = "ade_corpus_v2-classification ag_news amazon_polarity anli circa climate_fever dbpedia_14 discovery emo emotion ethos-directed_vs_generalized ethos-disability ethos-gender ethos-national_origin ethos-race ethos-religion ethos-sexual_orientation financial_phrasebank glue-cola glue-mnli glue-mrpc glue-qnli glue-qqp glue-rte glue-sst2 glue-wnli google_wellformed_query hate_speech18 hate_speech_offensive hatexplain imdb kilt_fever liar medical_questions_pairs onestop_english paws poem_sentiment rotten_tomatoes scicite scitail sick sms_spam superglue-cb superglue-rte superglue-wic superglue-wsc tab_fact trec trec-finegrained tweet_eval-emoji tweet_eval-emotion tweet_eval-hate tweet_eval-irony tweet_eval-offensive tweet_eval-sentiment tweet_eval-stance_abortion tweet_eval-stance_atheism tweet_eval-stance_climate tweet_eval-stance_feminist tweet_eval-stance_hillary wiki_qa yahoo_answers_topics yelp_polarity".split()

with open("cases/support.json", "w") as f_w:
    cnt = 0
    for t in all_tasks:
        support = os.path.join("raw_data", "gym", t)
        files = sorted(os.listdir(support))
        for f in files:
            if f.endswith("_support_shot_0.json"):
                support_t = os.path.join(support, f)
                cnt += 1
                with open(support_t, "r") as f_r:
                    f_w.write(f_r.read())
                print(f)
    print("all tasks ", len(all_tasks))
    print(cnt)
