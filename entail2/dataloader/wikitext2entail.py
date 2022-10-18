from itertools import islice
from typing import Dict, List, Generator
from collections import defaultdict
from pprint import pprint
from random import sample
from nltk.corpus import wordnet
from entail2.common.datastructure import QCH, Sentence, CH_Label, QCH_sampler
from entail2.common.textprocessing import simple_tokenize
from entail2.dataloader.base import Map_CH_dataset, dataset2loader

from torch.utils.data import DataLoader
import torch

import os
import csv


RAWROOT = "raw_data/oneshot_wikitext2"
TRAIN = os.path.join(RAWROOT, "train.csv")
TEST = os.path.join(RAWROOT, "test.csv")


def read_dataset(datasplit: str = "train"):
    assert datasplit in ("train", "test")
    if datasplit == "train":
        path = TRAIN
    else:
        path = TEST

    with open(path, "r") as f:
        csvreader = csv.DictReader(f)
        for row in csvreader:
            # TODO replace word label with wordnet difinition
            yield CH_Label(
                context=Sentence(
                    tokens=simple_tokenize(row["sentence"]), terms=[row["label"]]
                ),
                hypothesis=Sentence(tokens=simple_tokenize(row["label"])),
                mlabel=row["label"],
                meta=None,
            )


def gen_all_wikitext_QCH(datasplit: str = "train"):
    assert datasplit in ("train", "test")
    if datasplit == "train":
        path = TRAIN
    else:
        path = TEST

    label_dict = defaultdict(list)
    for chl in read_dataset(path):
        label_dict[chl.mlabel].append(chl)
    while True:
        yield from QCH_sampler(label_dict)


# class WikitextDataset(Iter_dataset):
#     def __init__(self, split, weight):
#         super(WikitextDataset, self).__init__(gen_all_wikitext_QCH, split, weight)


class WikitextDataset(Map_CH_dataset):
    def __init__(self, split, weight):
        super(WikitextDataset, self).__init__(read_dataset, split, weight)


def wikitext(batch_sz):
    train_dataset = WikitextDataset("train", None)
    test_dataset = WikitextDataset("test", None)

    train_loader = dataset2loader(
        train_dataset,
        batch_sz,
        cls_per_batch=batch_sz // 4,
        collate_fn=train_dataset.collate_fn,
    )
    test_loader = dataset2loader(
        test_dataset,
        batch_sz,
        cls_per_batch=batch_sz // 4,
        collate_fn=test_dataset.collate_fn,
    )

    return train_loader, test_loader


if __name__ == "__main__":
    # wikitext = WikitextDataset("train", None)

    # for qch_batch in wikitext:
    #     pprint(qch_batch)

    train, _ = wikitext(32)
    for batch in train:
        pprint(batch)
        break
