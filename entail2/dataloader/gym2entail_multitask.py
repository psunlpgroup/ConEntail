import os
import json
import re
import string
import random
import logging
import numpy as np
import argparse
from pprint import pprint
from collections import Counter, defaultdict
import string

from tqdm import tqdm
from itertools import chain
import torch
from torch.utils.data import (
    Dataset,
    TensorDataset,
    DataLoader,
    RandomSampler,
    SequentialSampler,
)


from entail2.common.textprocessing import simple_tokenize

from entail2.dataloader.base import Map_CH_dataset, dataset2loader
from entail2.common.datastructure import QCH, Sentence, CH_Label, QCH_sampler
from collections import defaultdict

RAWROOT = "raw_data/gym"
TRAIN = os.path.join(RAWROOT, "train-multi-ufsl_tasks-128.json")
DEV = os.path.join(RAWROOT, "dev-multi-ufsl_tasks.json")
TEST = os.path.join(RAWROOT, "test-multi-ufsl_tasks.json")

LABEL_DIC = os.path.join(RAWROOT, "label_dic.json")


def get_label_dic():
    if os.path.exists(LABEL_DIC):
        return json.load(open(LABEL_DIC, "r"))
    else:
        label_dic = defaultdict(set)
        # with open(TRAIN, "r") as f_train, open(DEV, "r") as f_dev, open(
        #     TEST, "r"
        # ) as f_test:

        #     for row in chain(f_train, f_dev, f_test):
        with open(TRAIN, "r") as f_train:

            for row in chain(f_train):
                row = json.loads(row)
                label_dic[row["task_name"]].add(row["out"])
        label_dic = {k: list(v) for k, v in label_dic.items()}
        json.dump(label_dic, open(LABEL_DIC, "w"), indent=4)

        return label_dic


def context_to_multichoice(context, all_labels):
    context = context.replace("\n", " ").replace("\t", " ")

    final_text = ""
    for i, e in enumerate(all_labels):
        final_text += f" (i) {e} "
    final_text += f" \\n {context}"
    return final_text


def read_dataset(datasplit: str = "train", training_shots: int = 128):

    label_dic = get_label_dic()

    assert datasplit in ("train", "test")
    if datasplit == "train":
        path = os.path.join(
            RAWROOT, "train-multi-ufsl_tasks-" + str(training_shots) + ".json"
        )
        # path = TRAIN
    elif datasplit == "test":
        path = DEV
    else:
        path = NotImplementedError

    with open(path, "r") as f:
        # if efl_neg == False:
        for row in f:
            row = json.loads(row)
            yield CH_Label(
                context=Sentence(tokens=simple_tokenize(row["in"]), terms=[]),
                hypothesis=Sentence(tokens=simple_tokenize(row["out"])),
                multichoice_context=Sentence(
                    tokens=simple_tokenize(
                        context_to_multichoice(row["in"], label_dic[row["task_name"]])
                    ),
                    terms=[],
                ),
                mlabel=str(row["task_name"]) + "_" + row["out"],
                meta=row["task_name"],
            )


class GymDataset(Map_CH_dataset):
    def __init__(self, split, training_shots, tokenizer):
        super(GymDataset, self).__init__(
            read_dataset, split, training_shots, tokenizer=tokenizer
        )


def gym(batch_sz, tokenizer, use_sampler=True, training_shots=128):
    train_dataset = GymDataset("train", training_shots, tokenizer)
    # dev_dataset = GymDataset("test", None, tokenizer)
    # test_dataset = WikitextDataset("test", None)

    train_loader = dataset2loader(
        train_dataset,
        batch_sz,
        cls_per_batch=5,
        collate_fn=train_dataset.collate_fn,
        meta_per_batch=1,
        use_sampler=use_sampler,
    )
    # dev_loader = dataset2loader(
    #     dev_dataset,
    #     batch_sz,
    #     cls_per_batch=5,
    #     collate_fn=dev_dataset.collate_fn,
    #     meta_per_batch=1,
    #     use_sampler=use_sampler
    # )

    return train_loader, None


def get_tasks_list(filename, split_name):
    with open(filename, "r") as fin:
        split_dict = json.load(fin)
    return split_dict[split_name]


class NLPFewshotGymMultiTaskData(object):
    def __init__(
        self, logger, args, data_path, tasks, data_split, data_type, is_training
    ):
        self.data_path = data_path
        self.data_type = data_type
        self.training_shots = args.training_shots

        self.data = []

        # keep "sorted" so that things are consistent across machines
        for task in sorted(tasks):
            task_dir = os.path.join(self.data_path, task)
            if not os.path.exists(task_dir):
                print("Task directory %s does not exist, please try to download the dataset individually", task_dir)
                continue
            files = sorted(os.listdir(task_dir))
            prefixes = []
            for filename in files:
                if not filename.endswith(".tsv"):
                    continue
                prefix = "_".join(filename.split("_")[:-1])
                if prefix not in prefixes:
                    prefixes.append(prefix)

            for prefix in prefixes:
                with open(os.path.join(task_dir, prefix + "_train.tsv")) as fin:
                    lines = fin.readlines()

                train_examples = []
                for i, line in enumerate(lines):
                    d = line.strip().split("\t")
                    d_head = "\t".join(d[:-1])
                    d_last = d[-1]
                    train_examples.append((d_head, d_last))

                # with open(os.path.join(task_dir, prefix + "_dev.tsv")) as fin:
                #     lines = fin.readlines()

                dev_examples = []
                # for line in lines:
                #     d = line.strip().split("\t")
                #     d_head = "\t".join(d[:-1])
                #     d_last = d[-1]
                #     dev_examples.append((d_head, d_last))

                self.data.append(
                    {
                        "task_name": task,
                        "task_prefix": prefix,
                        "train_examples": train_examples,
                        "dev_examples": dev_examples,
                    }
                )

        self.data_split = data_split
        self.is_training = is_training
        self.logger = logger
        self.args = args

        self.metric = "EM"
        self.tokenizer = None
        self.dataset = None
        self.dataloader = None
        self.cache = None

        self.load = not args.debug

        self.gen_early_stop = False

    def __len__(self):
        return len(self.data)

    def flatten(self, answers):
        new_answers, metadata = [], []
        for answer in answers:
            metadata.append((len(new_answers), len(new_answers) + len(answer)))
            new_answers += answer
        return new_answers, metadata

    def load_dataset(self, do_return=False):
        # self.tokenizer = tokenizer
        # postfix = tokenizer.__class__.__name__.replace("zer", "zed")
        split_identifier = self.args.custom_tasks_splits.split("/")[-1]
        if split_identifier.endswith(".json"):
            split_identifier = split_identifier[:-5]

        # preprocessed_path = os.path.join(
        #     self.data_path,
        #     self.data_type + "-multi-{}.json".format(split_identifier),
        # )
        preprocessed_path = os.path.join(
            self.data_path,
            self.data_type
            + "-multi-{}-{}.json".format(split_identifier, self.training_shots),
        )
        self.preprocessed_path = preprocessed_path
        if self.load and os.path.exists(preprocessed_path):
            # load preprocessed input
            self.logger.info("Load preprocessed data from {}".format(preprocessed_path))
        else:
            self.logger.info("Dump preprocessed data to {}".format(preprocessed_path))
            with open(preprocessed_path, "w") as f:
                inputs = []
                outputs = []
                cnt = Counter()
                for task in self.data:
                    task_name = task["task_name"]
                    if self.data_split == "train" or self.data_split == "all":
                        for dp in task["train_examples"]:
                            if cnt[dp[1]] >= self.training_shots:
                                continue
                            else:

                                f.write(
                                    json.dumps(
                                        {
                                            "in": dp[0],
                                            "out": dp[1],
                                            "task_name": task_name,
                                        }
                                    )
                                )
                                f.write("\n")
                                cnt[dp[1]] += 1

                            inputs.append(" [{}] {}".format(task_name, dp[0]))
                            outputs.append(item for item in dp[1])
                    if self.data_split == "dev" or self.data_split == "all":
                        for dp in task["dev_examples"]:
                            f.write(
                                json.dumps(
                                    {"in": dp[0], "out": dp[1], "task_name": task_name}
                                )
                            )
                            f.write("\n")
                            inputs.append(" [{}] {}".format(task_name, dp[0]))
                            outputs.append(item for item in dp[1])
                    task_length = len(task["train_examples"]) + len(
                        task["dev_examples"]
                    )
                    self.logger.info("{}: {}".format(task_name, task_length))


def pprint_data(split):
    for i, line in enumerate(read_dataset(split, efl_neg=True)):
        pprint(line)
        if i > 20:
            break


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", action="store_false", help="Use a subset of data for debugging"
    )
    parser.add_argument("--do_lowercase", action="store_true", default=False)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_output_length", type=int, default=64)

    parser.add_argument("--append_another_bos", action="store_true", default=False)
    parser.add_argument(
        "--custom_tasks_splits", type=str, default="entail2/dataloader/ufsl_tasks.json"
    )
    parser.add_argument("--output_dir", default="logs", type=str)
    # parser.add_argument("--train_dir", default="raw_data/gym")
    parser.add_argument("--train_dir", default="raw_data/gym")
    parser.add_argument("--do_train", action="store_true")
    # parser.add_argument("--model", default="bert-base-uncased", required=False)
    parser.add_argument("--training_shots", type=int, default=128)

    args = parser.parse_args()
    gym_all_tasks = get_tasks_list(args.custom_tasks_splits, "all")
    gym_train_tasks = get_tasks_list(args.custom_tasks_splits, "train")
    gym_dev_tasks = get_tasks_list(args.custom_tasks_splits, "dev")
    gym_test_tasks = get_tasks_list(args.custom_tasks_splits, "test")

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
    # logger.info(args)
    # logger.info(args.output_dir)
    # tokenizer = BartTokenizer.from_pretrained(args.model)
    train_data = NLPFewshotGymMultiTaskData(
        logger,
        args,
        args.train_dir,
        tasks=gym_train_tasks,
        data_split="train",
        data_type="train",
        is_training=True,
    )
    train_data.load_dataset()
    dev_data = NLPFewshotGymMultiTaskData(
        logger,
        args,
        args.train_dir,
        tasks=gym_dev_tasks,
        data_split="train",
        data_type="dev",
        is_training=True,
    )
    dev_data.load_dataset()
    test_data = NLPFewshotGymMultiTaskData(
        logger,
        args,
        args.train_dir,
        tasks=gym_test_tasks,
        data_split="train",
        data_type="test",
        is_training=True,
    )
    test_data.load_dataset()
    # train, dev = gym(32, tok...)
    # print(dev_data.preprocessed_path)
    # for batch in dev:
    #     print(batch.keys())
    #     pprint(batch["meta"].size())
    #     pprint(batch["mlabel"].size())
    #     pprint(batch["meta"])
    #     pprint(batch["mlabel"])
    #     print(train.dataset.meta2idx)
    #     break


if __name__ == "__main__":
    # pprint_data("train")
    main()
