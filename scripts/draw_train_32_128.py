import json
import argparse
import os

import numpy as np
from pprint import pprint
from collections import defaultdict
from entail2.dataloader.gym2entail_multitask import get_tasks_list


def main(args):

    # models = ["crossfit", "unifew", "efl", "efl_multichoice", "entail2"]
    # models = ["entail2", "efl", "crossfit", "unifew"]
    # models = ["unifew", "crossfit"]
    # shots = [5] + list(range(10, 110, 10))
    shots = "32 48 64 80 96 112".split()
    print(shots)
    # models = ["entail2", "efl_no_cl", "unifew", "crossfit"]
    models = ["entail2", "efl_no_cl"]
    # models = ["efl_multichoice"]
    # models = ["entail2"]
    # models = ["unifew"]

    # gym_test_tasks = get_tasks_list(args.custom_tasks_splits, "new_test")

    # gym_test_tasks = "glue-cola glue-qqp glue-sst2 glue-mrpc scitail amazon_polarity ag_news rotten_tomatoes hate_speech_offensive imdb".split(
    #     " "
    # )

    # gym_test_tasks = "glue-cola amazon_polarity rotten_tomatoes".split(" ")
    # gym_test_tasks = (
    #     "glue-sst2 glue-mrpc scitail ag_news hate_speech_offensive".split(" ")
    # )
    gym_test_tasks = "glue-cola amazon_polarity rotten_tomatoes glue-sst2 glue-mrpc scitail ag_news hate_speech_offensive glue-qqp".split(
        " "
    )
    # gym_test_tasks = "glue-qqp glue-sst2 glue-mrpc scitail ag_news hate_speech_offensive".split(" ")

    # gym_test_tasks = ["rotten_tomatoes","hate_speech_offensive", "imdb"]

    for model_name in models:
        # print("=" * 20)
        # print(model_name)

        pred = defaultdict(list)
        base = defaultdict(list)
        for task in gym_test_tasks:
            for shot in shots:
                result_path = os.path.join(
                    args.data_dir,
                    task,
                    model_name
                    + "test_shots_1"
                    + "training_shots_"
                    + str(shot)
                    + "_result.json",
                )
                try:
                    with open(result_path, "r") as f:
                        times_results = []
                        for i, line in enumerate(f):
                            if i > 2:
                                break
                            dic = json.loads(line)
                            times_results.append(dic["acc"]["pred"])

                    pred[task].append(times_results)
                except:
                    print(result_path)
        print(model_name)
        print(pred)
        # score_str = model_name
        # avg_scores = []
        # for t in gym_test_tasks:
        #     score_str += "\t&\t\\res{%.1f}{%.2f} " % (np.mean(pred[t]) * 100, np.std(pred[t]) * 100)
        #     # score_str += "\t&\t\\mathnum{%.1f} " % (np.mean(pred[t]) * 100)
        #     avg_scores.append(np.mean(pred[t]))
        # score_str += "\t&\t\\mathnum{%.1f} " % (np.mean(avg_scores) * 100)
        # score_str += " \\\\"

        # print(score_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model_name_list = ["entail2", "efl", "crossfit", "unifew"]

    # parser.add_argument("--model_name", choices=model_name_list, default="entail2")

    parser.add_argument("--data_dir", default="raw_data/gym", required=False)
    parser.add_argument(
        "--custom_tasks_splits", type=str, default="entail2/dataloader/ufsl_tasks.json"
    )
    parser.add_argument(
        "--test_shots", default=10, type=int, help="shot number on meta-test phase"
    )
    parser.add_argument(
        "--test_times",
        default=3,
        type=int,
        help="test times on different support sets",
    )
    args = parser.parse_args()
    main(args)
