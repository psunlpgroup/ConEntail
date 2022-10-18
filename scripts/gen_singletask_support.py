import os
import json
import re
import string
import random

import numpy as np

from collections import Counter, defaultdict
from tqdm import tqdm

import torch
from torch.utils.data import (
    Dataset,
    TensorDataset,
    DataLoader,
    RandomSampler,
    SequentialSampler,
)
import argparse
import logging


class NLPFewshotGymSingleTaskData(object):
    def __init__(self, logger, args, data_path, data_type, is_training):
        # should give the tasks used in this split in the var "tasks"
        self.data_path = data_path
        self.data_type = data_type

        self.data = []

        self.task_name = "_".join(self.data_path.split("/")[-1].split("_")[:-3])

        with open(data_path) as fin:
            lines = fin.readlines()

        # train_examples = []
        for line in lines:
            d = line.strip().split("\t")
            self.data.append((d[0], d[1:]))

        self.is_training = is_training
        self.load = not args.debug
        self.logger = logger
        self.args = args

        self.shots = args.shots

        # self.metric = METRICS[self.task_name]
        # self.max_input_length = self.args.max_input_length
        self.tokenizer = None
        self.dataset = None
        self.dataloader = None
        self.cache = None

        self.gen_early_stop = False

    def __len__(self):
        return len(self.data)

    def flatten(self, answers):
        new_answers, metadata = [], []
        for answer in answers:
            metadata.append((len(new_answers), len(new_answers) + len(answer)))
            new_answers += answer
        return new_answers, metadata

    def load_dataset(self, shots=1, seed=0, do_return=False):
        postfix = "shot_" + str(shots) + "_seed_" + str(seed)
        preprocessed_path = os.path.join(
            "/".join(self.data_path.split("/")[:-1]),
            self.data_path.split("/")[-1]
            .replace("train", "support")
            .replace(".tsv", "_{}.json".format(postfix)),
        )

        self.preprocessed_path = preprocessed_path
        if self.load and os.path.exists(preprocessed_path):
            # load preprocessed input
            self.logger.info(
                "Loading pre-tokenized data from {}".format(preprocessed_path)
            )
            print("err")
            exit()
        else:

            self.logger.info("Dump preprocessed data to {}".format(preprocessed_path))
            with open(preprocessed_path, "w") as f:
                inputs = []
                outputs = []
                way_shot_dic = defaultdict(list)
                for dp in self.data:
                    # print(dp)
                    way_shot_dic[dp[1][0]].append(dp[0])
                for way in way_shot_dic:
                    np.random.shuffle(way_shot_dic[way])
                    for s in range(shots):

                        inputs.append(way_shot_dic[way][s])
                        outputs.append([way])
                        f.write(
                            json.dumps(
                                {
                                    "in": way_shot_dic[way][s],
                                    # "in": "null",
                                    "out": [way],
                                    "task_name": self.task_name,
                                }
                            )
                        )
                        f.write("\n")

            self.logger.info("Printing 3 examples")
            for i in range(len(inputs)):
                self.logger.info(inputs[i])
                self.logger.info(outputs[i])


def main(args):

    # gym_all_tasks = get_tasks_list(args.custom_tasks_splits, "all")
    # gym_train_tasks = get_tasks_list(args.custom_tasks_splits, "train")
    # gym_dev_tasks = get_tasks_list(args.custom_tasks_splits, "dev")
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    log_filename = "{}log.txt".format("" if args.do_train else "eval_")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, log_filename)),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    join_dir = os.path.join(args.data_dir, args.task_dir)

    files = sorted(os.listdir(join_dir))
    prefixes = []
    for filename in files:
        if not filename.endswith(".tsv"):
            continue
        prefix = "_".join(filename.split("_")[:-1])
        if prefix not in prefixes:
            prefixes.append(prefix)
            break

    for prefix in prefixes:

        support_path = os.path.join(join_dir, prefix + "_train.tsv")

    test_data = NLPFewshotGymSingleTaskData(
        logger, args, support_path, data_type="support", is_training=False
    )

    for s in range(args.times):
        test_data.load_dataset(shots=args.shots, seed=s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_dir", default="glue-sst2", required=False)
    parser.add_argument("--data_dir", default="raw_data/gym", required=False)
    parser.add_argument(
        "--debug", action="store_false", help="Use a subset of data for debugging"
    )
    parser.add_argument("--do_lowercase", action="store_true", default=False)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_output_length", type=int, default=64)
    parser.add_argument("--append_another_bos", action="store_true", default=False)
    parser.add_argument(
        "--custom_tasks_splits", type=str, default="entail2/dataloader/gym_tasks.json"
    )
    parser.add_argument("--output_dir", default="logs", type=str)
    parser.add_argument("--train_dir", default="raw_data/gym")
    parser.add_argument("--do_train", action="store_true")
    # parser.add_argument("--model", default="facebook/bart-base", required=False)
    # parser.add_argument("--test_file", default="raw_data/gym/glue-sst2", required=False)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--times", type=int, default=20)
    # parser.add_argument("--seed", type=int, default=20)
    args = parser.parse_args()
    main(args)
