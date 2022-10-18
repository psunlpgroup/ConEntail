import json
import argparse
import os

import numpy as np
from pprint import pprint
from collections import defaultdict
from entail2.dataloader.gym2entail_multitask import get_tasks_list


def main(args):

    models = ["crossfit", "unifew", "efl", "efl_multichoice", "entail2"]
    models = ["efl_no_cl"]
    models = ["entail2", "efl_no_cl", "crossfit", "unifew"]
    # models = ["unifew", "crossfit"]
    # models = ["entail2"]
    # models = ["efl_multichoice"]
    # models = ["entail2"]
    # models = ["unifew"]

    # gym_test_tasks = get_tasks_list(args.custom_tasks_splits, "new_test")

    # gym_test_tasks = "glue-cola glue-qqp glue-sst2 glue-mrpc scitail amazon_polarity ag_news rotten_tomatoes hate_speech_offensive imdb".split(
    #     " "
    # )

    gym_test_tasks = "glue-cola glue-qqp glue-sst2 glue-mrpc scitail amazon_polarity ag_news rotten_tomatoes hate_speech_offensive".split(
        " "
    )
    # gym_test_tasks = "glue-cola glue-mrpc scitail amazon_polarity ag_news rotten_tomatoes hate_speech_offensive".split(
    #     " "
    # )
    # gym_test_tasks = ["rotten_tomatoes","hate_speech_offensive", "imdb"]

    for model_name in models:
        # print("=" * 20)
        # print(model_name)

        pred = defaultdict(list)
        base = defaultdict(list)
        for task in gym_test_tasks:
            # result_path = os.path.join(
            #     args.data_dir,
            #     task,
            #     model_name + "_shots_" + str(args.test_shots) + "_result.json",
            # )
            result_path = os.path.join(
                args.data_dir,
                task,
                model_name + "_shots_" + str(args.test_shots) + "_finetune_result.json",
            )

            with open(result_path, "r") as f:
                json_str = ""
                for i, line in enumerate(f, start=1):
                    json_str += "\n" + line
                    # if i % 18 == 0 and i != 0:
                    dic = json.loads(line)
                    json_str = ""

                    pred[task].append(dic["acc"]["pred"])
                    base[task].append(dic["acc"]["base"])
                    # print(task, dic["acc"]["pred"], dic["acc"]["base"])
        # pprint(pred)
        # pprint(base)
        for t in gym_test_tasks:
            # print(
            #     "pred\t%s\t%.1f ($\\pm$%.2f) "
            #     % (t, np.mean(pred[t]) * 100, np.std(pred[t]) * 100)
            # )
            pass
        score_str = model_name
        avg_scores = []
        for t in gym_test_tasks:
            score_str += "\t&\t\\res{%.1f}{%.1f} " % (
                np.mean(pred[t]) * 100,
                np.std(pred[t]) * 100,
            )
            # score_str += "\t&\t\\mathnum{%.1f} " % (np.mean(pred[t]) * 100)
            avg_scores.append(np.mean(pred[t]))
        score_str += "\t&\t\\mathnum{%.1f} " % (np.mean(avg_scores) * 100)
        score_str += " \\\\"

        print(score_str)


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
        default=10,
        type=int,
        help="test times on different support sets",
    )
    args = parser.parse_args()
    main(args)
