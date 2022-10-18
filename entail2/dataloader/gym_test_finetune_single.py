import os
import json
import re
import string
import random
import logging
import numpy as np
import argparse
from pprint import pprint
from collections import Counter
from tqdm import tqdm

import torch
from torch.utils.data import (
    Dataset,
    TensorDataset,
    DataLoader,
    RandomSampler,
    SequentialSampler,
)


from entail2.common.textprocessing import simple_tokenize

from entail2.dataloader.base import Map_CH_dataset, dataset2loader, Map_QCH_dataset
from entail2.common.datastructure import QCH, Sentence, CH_Label, QCH_sampler
from collections import defaultdict

RAWROOT = "raw_data/gym"
TRAIN = os.path.join(RAWROOT, "train-multi-ufsl_tasks.json")
DEV = os.path.join(RAWROOT, "dev-multi-ufsl_tasks.json")
TEST = os.path.join(RAWROOT, "test-multi-ufsl_tasks.json")
LABEL_DIC = os.path.join(RAWROOT, "label_dic.json")

# TEST = os.path.join(RAWROOT, "test.csv")


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


def context_to_multichoice(context, all_labels):
    context = context.replace("\n", " ").replace("\t", " ")

    final_text = ""
    for i, e in enumerate(all_labels):
        final_text += f" (i) {e} "
    final_text += f" \\n {context}"
    return final_text


def read_finetune(support_path):
    def read_dataset(datasplit: str = "test", training_shots: int = 128):
        label_dic = get_label_dic()

        with open(support_path, "r") as support_f:
            # support_list = []
            for shot in support_f:
                shot_row = json.loads(shot)
                context = Sentence(tokens=simple_tokenize(shot_row["in"]), terms=[])
                hypo = Sentence(tokens=simple_tokenize(shot_row["out"][0]), terms=[])
                yield CH_Label(
                    context=context,
                    hypothesis=hypo,
                    multichoice_context=Sentence(
                        tokens=simple_tokenize(
                            context_to_multichoice(
                                shot_row["in"], label_dic[shot_row["task_name"]]
                            )
                        ),
                        terms=[],
                    ),
                    mlabel=str(shot_row["task_name"]) + "_" + shot_row["out"][0],
                    meta=shot_row["task_name"],
                )

    return read_dataset


def read_efl_finetune(support_path):
    def read_dataset(datasplit: str = "test", training_shots: int = 128):
        label_dic = get_label_dic()

        with open(support_path, "r") as support_f:
            # support_list = []
            for shot in support_f:
                shot_row = json.loads(shot)
                label = random.choice(label_dic[shot_row["task_name"]])
                context = Sentence(
                    tokens=simple_tokenize(shot_row["in"])
                    + ["[SEP]"]
                    + simple_tokenize(label),
                    terms=[],
                )
                hypo = Sentence(tokens=simple_tokenize(shot_row["out"][0]), terms=[])
                yield CH_Label(
                    context=context,
                    hypothesis=hypo,
                    multichoice_context=Sentence(
                        tokens=simple_tokenize(
                            context_to_multichoice(
                                shot_row["in"], label_dic[shot_row["task_name"]]
                            )
                        ),
                        terms=[],
                    ),
                    mlabel=shot_row["out"][0] == label,
                    # mlabel=str(shot_row["task_name"]) + "_" + shot_row["out"][0],
                    meta=shot_row["task_name"],
                )

    return read_dataset


class GymFinetuneSingleDataset(Map_CH_dataset):
    def __init__(self, support_path, tokenizer):

        super(GymFinetuneSingleDataset, self).__init__(
            read_finetune(support_path), 128, tokenizer=tokenizer
        )


class EFLGymFinetuneSingleDataset(Map_CH_dataset):
    def __init__(self, support_path, tokenizer):

        super(EFLGymFinetuneSingleDataset, self).__init__(
            read_efl_finetune(support_path), 128, tokenizer=tokenizer
        )


def gym_finetune(support_path, tokenizer):

    support_dataset = GymFinetuneSingleDataset(support_path, tokenizer)
    batch_sz = len(support_dataset) if len(support_dataset) <= 32 else 32
    support_loader = dataset2loader(
        support_dataset,
        batch_sz,
        use_sampler=False,
        collate_fn=support_dataset.collate_fn,
    )

    return support_loader, support_loader


def gym_efl_finetune(support_path, tokenizer):

    support_dataset = EFLGymFinetuneSingleDataset(support_path, tokenizer)
    batch_sz = len(support_dataset) if len(support_dataset) <= 32 else 32
    support_loader = dataset2loader(
        support_dataset,
        batch_sz,
        use_sampler=False,
        collate_fn=support_dataset.collate_fn,
    )
    # print("new_loader")
    return support_loader, support_loader


# if __name__ == "__main__":
#     test_sst2 = GymTestSingleDataset(
#         support_path="raw_data/gym/glue-sst2/glue-sst2_128_100_support-BartTokenized.json",
#         test_path="raw_data/gym/glue-sst2/glue-sst2_128_100_test-BartTokenized.json",
#     )
#     for i, item in enumerate(test_sst2):
#         pprint(item)
#         if i > 20:
#             break
