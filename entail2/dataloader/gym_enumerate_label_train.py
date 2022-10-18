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


from entail2.common.textprocessing import simple_tokenize

from entail2.dataloader.base import Map_CH_dataset, dataset2loader
from entail2.common.datastructure import QCH, Sentence, CH_Label, QCH_sampler
from collections import defaultdict

RAWROOT = "raw_data/gym"
TRAIN = os.path.join(RAWROOT, "train-multi-ufsl_tasks.json")
DEV = os.path.join(RAWROOT, "dev-multi-ufsl_tasks.json")
TEST = os.path.join(RAWROOT, "test-multi-ufsl_tasks.json")

LABEL_DIC = os.path.join(RAWROOT, "label_dic.json")


def context_to_multichoice(context, all_labels):
    context = context.replace("\n", " ").replace("\t", " ")

    final_text = ""
    for i, e in enumerate(all_labels):
        final_text += f" (i) {e} "
    final_text += f" \\n {context}"
    return final_text


def get_label_dic():
    if os.path.exists(LABEL_DIC):
        return json.load(open(LABEL_DIC, "r"))
    else:
        label_dic = defaultdict(set)
        with open(TRAIN, "r") as f_train, open(DEV, "r") as f_dev, open(
            TEST, "r"
        ) as f_test:

            for row in chain(f_train, f_dev, f_test):
                row = json.loads(row)
                label_dic[row["task_name"]].add(row["out"])
        label_dic = {k: list(v) for k, v in label_dic.items()}
        json.dump(label_dic, open(LABEL_DIC, "w"), indent=4)

        return label_dic


def read_dataset_and_enumerate_label(
    datasplit: str = "train", training_shots: int = 128
):
    label_dic = get_label_dic()

    assert datasplit in ("train", "test")
    if datasplit == "train":
        path = os.path.join(
            RAWROOT, "train-multi-ufsl_tasks-" + str(training_shots) + ".json"
        )
    elif datasplit == "test":
        path = DEV
    else:
        path = NotImplementedError

    with open(path, "r") as f:
        for row in f:
            row = json.loads(row)
            label = random.choice(label_dic[row["task_name"]])

            yield CH_Label(
                context=Sentence(
                    tokens=simple_tokenize(row["in"])
                    + ["[SEP]"]
                    + simple_tokenize(label),
                    terms=[],
                ),
                hypothesis=Sentence(tokens=simple_tokenize(row["out"])),
                multichoice_context=Sentence(
                    tokens=simple_tokenize(
                        context_to_multichoice(row["in"], label_dic[row["task_name"]])
                    ),
                    terms=[],
                ),
                mlabel=row["out"] == label,
                meta=row["task_name"],
            )


class GymEFL(Map_CH_dataset):
    def __init__(self, split, training_shots, tokenizer):
        super(GymEFL, self).__init__(
            read_dataset_and_enumerate_label, split, training_shots, tokenizer=tokenizer
        )


def gym_efl(batch_sz, tokenizer, use_sampler=True, training_shots=128):
    train_dataset = GymEFL("train", training_shots, tokenizer)
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


def pprint_data(split):
    for i, line in enumerate(read_dataset_and_enumerate_label(split)):
        pprint(line)
        if i > 20:
            break


if __name__ == "__main__":
    pprint_data("train")
