from itertools import islice
from typing import Generator, Iterator, List, Optional, Dict, Any
from collections import defaultdict
from functools import partial
from pprint import pprint
from random import sample
from nltk.corpus import wordnet
from entail2.common.datastructure import QCH, Sentence, CH_Label, QCH_sampler
from entail2.common.textprocessing import simple_tokenize
from entail2.dataloader.base import (
    Map_QCH_dataset,
    Map_CH_dataset,
    dataset2loader,
)
from torch.utils.data import Dataset

from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch

import os
import json


RAWROOT = "raw_data/fewrel2"
TRAIN = os.path.join(RAWROOT, "train_wiki.json")
VAL = os.path.join(RAWROOT, "val_wiki.json")

TESTROOT = "raw_data/fewrel_test"
ID2NAME = os.path.join(RAWROOT, "pid2name.json")


def read_train_dataset(datasplit: str = "train", UNKNOWN_RELATION: bool = False):
    assert datasplit in ("train", "test")
    if datasplit == "train":
        path = TRAIN
    else:
        path = VAL

    with open(path, "r") as f, open(ID2NAME, "r") as id_f:
        json_data = json.load(f)
        id2name = json.load(id_f)
        classes = list(json_data.keys())
        for r, rows in json_data.items():
            r_name, r_discription = id2name[r]

            if UNKNOWN_RELATION:
                r_name = "unknown"

            for row in rows:
                h, t = simple_tokenize(row["h"][0]), simple_tokenize(row["t"][0])
                hypo = h + simple_tokenize("subject " + r_name + " object") + t
                yield CH_Label(
                    context=Sentence(tokens=row["tokens"], terms=h + t),
                    hypothesis=Sentence(tokens=hypo),
                    mlabel=r,
                    meta=None,
                )


def read_test_dataset(
    datasplit: str = "test",
    name="test_wiki_input-5-5-0.15.json",
    UNKNOWN_RELATION: bool = False,
):
    assert datasplit == "test"

    path = os.path.join(TESTROOT, name)

    with open(path, "r") as f, open(ID2NAME, "r") as id_f:
        json_data = json.load(f)
        id2name = json.load(id_f)
        for rows in json_data:
            ins = []
            query = rows["meta_test"]

            supports = rows["meta_train"]
            relation_supports = rows["relation"]
            # print(simple_tokenize(query["h"][0]), simple_tokenize(query["t"][0])))
            query_sent = Sentence(
                query["tokens"],
                terms=simple_tokenize(query["h"][0]) + simple_tokenize(query["t"][0]),
            )
            for i, sup in enumerate(supports):
                for shot in sup:
                    if UNKNOWN_RELATION:
                        r_name = "unknown"
                    else:
                        r_name, r_discription = id2name[relation_supports[i]]
                    h, t = simple_tokenize(shot["h"][0]), simple_tokenize(shot["t"][0])
                    hypo = h + simple_tokenize("subject " + r_name + " object") + t
                    ins.append(
                        QCH(
                            query=query_sent,
                            context=Sentence(tokens=shot["tokens"], terms=h + t),
                            hypothesis=Sentence(tokens=hypo, terms=h + t),
                            mlabel=r_name,
                        )
                    )
            yield ins


class FewrelUnknownRelationDataset(Map_CH_dataset):
    def __init__(self, split, weight):
        super(FewrelUnknownRelationDataset, self).__init__(
            partial(read_train_dataset, UNKNOWN_RELATION=True), split, weight
        )


class FewrelNameRelationDataset(Map_CH_dataset):
    def __init__(self, split, weight):
        super(FewrelNameRelationDataset, self).__init__(
            partial(read_train_dataset, UNKNOWN_RELATION=False), split, weight
        )


class FewrelTestNameRelationDataset(Map_QCH_dataset):
    def __init__(self, split, weight, name):
        super(FewrelTestNameRelationDataset, self).__init__(
            partial(read_test_dataset, name=name, UNKNOWN_RELATION=False), split, weight
        )


def fewrel(batch_sz):

    train_dataset = FewrelNameRelationDataset("train", None)
    test_dataset = FewrelNameRelationDataset("test", None)

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


def fewrel_submmit(batch_sz, name="test_wiki_input-5-5-0.15.json"):

    test_dataset = FewrelTestNameRelationDataset("test", None, name)

    test_loader = dataset2loader(
        test_dataset,
        batch_sz,
        use_sampler=False,
        cls_per_batch=batch_sz // 4,
        collate_fn=test_dataset.collate_fewrel_fn,
    )

    return test_loader, test_loader


if __name__ == "__main__":

    # _, test_loader = fewrel_submmit(1)
    # train_loader, dev_loader = fewrel(32)
    for i, item in enumerate(read_train_dataset("train")):
        pprint(item)
        if i > 20:
            break

    # for i, items in enumerate(
    #     FewrelTestNameRelationDataset(
    #         "test", None, name="test_wiki_input-5-5-0.15.json"
    #     )
    # ):
    #     for item in items:
    #         if item.mlabel == "employer" and (
    #             "New Yorker" in " ".join(item.context.tokens)
    #         ):
    #             pprint(item)
    #             print(item.mlabel)
    #             print()
