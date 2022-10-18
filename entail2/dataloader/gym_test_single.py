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
from transformers import BartTokenizer, BartConfig

# from transformers.tokenization_xlm import replace_unicode_punct

# from gym_utils import MyQADataset, MyDataLoader


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


def read_test(test_path, support_path):
    def read_dataset(datasplit: str = "test"):
        label_dic = get_label_dic()

        with open(test_path, "r") as test_f, open(support_path, "r") as support_f:
            support_list = []
            cnt = {}
            for shot in support_f:
                shot_row = json.loads(shot)
                context = Sentence(tokens=simple_tokenize(shot_row["in"]), terms=[])
                hypo = Sentence(tokens=simple_tokenize(shot_row["out"][0]), terms=[])
                meta = shot_row["out"]

                cnt[meta[0]] = cnt.get(meta[0], 0) + 1
                # if cnt[meta[0]] <= 10:
                if True:
                    support_list.append((context, hypo, meta))
            for row in test_f:
                ins = []
                row = json.loads(row)
                query = Sentence(tokens=simple_tokenize(row["in"]), terms=[])
                # meta=str(row["task_name"]) + "_" + row["out"]
                q_mlabel = row["out"]
                multichoice_query = Sentence(
                    tokens=simple_tokenize(
                        context_to_multichoice(
                            row["in"], label_dic.get(row["task_name"], ["unknown"])
                        )
                    ),
                    terms=[],
                )
                for i, (c, h, m) in enumerate(support_list):
                    query_hypo = query + h
                    ins.append(
                        QCH(
                            query=query,
                            context=c,
                            hypothesis=h,
                            multichoice_query=multichoice_query,
                            query_hypo=query_hypo,
                            mlabel=int(q_mlabel == m),
                            meta=m,
                        )
                    )
                yield ins

    return read_dataset


class GymTestSingleDataset(Map_QCH_dataset):
    def __init__(self, test_path, support_path, tokenizer):

        super(GymTestSingleDataset, self).__init__(
            read_test(test_path, support_path), "test", tokenizer=tokenizer
        )


def gym_test(batch_sz, test_path, support_path, tokenizer):

    test_dataset = GymTestSingleDataset(test_path, support_path, tokenizer)

    test_loader = dataset2loader(
        test_dataset,
        batch_sz,
        use_sampler=False,
        collate_fn=test_dataset.collate_gym_fn,
    )

    return test_loader, test_loader


# if __name__ == "__main__":
#     test_sst2 = GymTestSingleDataset(
#         support_path="raw_data/gym/glue-sst2/glue-sst2_128_100_support-BartTokenized.json",
#         test_path="raw_data/gym/glue-sst2/glue-sst2_128_100_test-BartTokenized.json",
#     )
#     for i, item in enumerate(test_sst2):
#         pprint(item)
#         if i > 20:
#             break
