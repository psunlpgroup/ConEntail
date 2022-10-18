# from torch.utils.data import IterableDataset, Dataloader
import torch
import random
from torch.utils.data import Dataset

# from torch.utils.data import IterableDataset, Dataset
from torch import Tensor
from torch.utils.data._utils.collate import default_collate
from transformers import AutoConfig, AutoModel, AutoTokenizer

from typing import Generator, Iterator, List, Optional, Dict, Any
from random import choices
from pprint import pprint

from itertools import cycle, chain
from collections import defaultdict, Counter
from entail2.common import QCH, CH_Label, sentence2bert, sentpair2bert
from entail2.common.datastructure import Sentence


class Map_QCH_dataset(Dataset):
    def __init__(
        self,
        gen,
        split,
        weight: Optional[int] = None,
        length: int = 128,
        tokenizer: Any = None,
    ):
        self.generator = gen
        self.split = split
        # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer = tokenizer
        self.length = length
        self.nested = True

        # self.mlabels = [d.mlabel for d in self.generator(self.split)]
        # self.m2idx = {m: i for i, m in enumerate(set(self.mlabels))}

        self.data = [d for d in self.generator(self.split)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index]

    def collate_gym_fn(self, data):
        support_len = len(data[0])
        data = [self.get_tensor(d) for ins in data for d in ins]
        data = default_collate(data)
        data = {k: torch.stack(v.split(support_len)) for k, v in data.items()}
        return data

    def collate_fewrel_fn(self, data):
        data = [self.get_tensor(d) for d in data[0]]
        data = default_collate(data)
        return data

    def collate_fn(self, data):
        data = [self.get_tensor(d) for d in data]
        data = default_collate(data)
        return data

    def get_tensor(self, qch: QCH) -> Dict[str, Any]:
        q, c, h, multi_q, qh, b, m, _ = qch
        q_bert, m_q = sentence2bert(q, self.tokenizer, self.length)

        qh_bert, m_qh = sentence2bert(qh, self.tokenizer, self.length)

        c_bert, m_c = sentence2bert(c, self.tokenizer, self.length)
        h_bert, m_h = sentence2bert(h, self.tokenizer, self.length)

        ch_bert, m_ch = sentpair2bert(c, h, self.tokenizer, self.length)
        multi_q_bert, m_multi_q = sentence2bert(multi_q, self.tokenizer, self.length)

        q_dic = {("q" + k): torch.squeeze(v, 0) for k, v in q_bert.items()}

        qh_dic = {("qh" + k): torch.squeeze(v, 0) for k, v in qh_bert.items()}

        c_dic = {("c" + k): torch.squeeze(v, 0) for k, v in c_bert.items()}
        h_dic = {("h" + k): torch.squeeze(v, 0) for k, v in h_bert.items()}

        ch_dic = {("ch" + k): torch.squeeze(v, 0) for k, v in ch_bert.items()}
        multi_q_dic = {
            ("mulit_q" + k): torch.squeeze(v, 0) for k, v in multi_q_bert.items()
        }
        marker_dic = {
            "m_q": m_q,
            "m_c": m_c,
            "m_h": m_h,
            "m_ch": m_ch,
            "m_multi_q": m_multi_q,
            "m_qh": m_qh,
        }
        # mdic = {"mlabel": torch.tensor(self.m2idx[m])}
        if type(m) is int:
            mdic = {"mlabel": torch.tensor(m)}
            return {
                **q_dic,
                **qh_dic,
                **c_dic,
                **h_dic,
                **ch_dic,
                **multi_q_dic,
                **marker_dic,
                **mdic,
            }
        else:
            return {
                **q_dic,
                **qh_dic,
                **c_dic,
                **h_dic,
                **ch_dic,
                **multi_q_dic,
                **marker_dic,
            }


class Map_CH_dataset(Dataset):
    def __init__(
        self,
        gen,
        split,
        training_shots: Optional[int] = 128,
        length: int = 128,
        tokenizer: Any = None,
    ):
        self.generator = gen
        self.training_shots = training_shots
        self.split = split
        self.tokenizer = tokenizer
        self.length = length

        self.mlabels = [
            d.mlabel for d in self.generator(self.split, self.training_shots)
        ]
        self.meta_mlabel_dic = defaultdict(set)

        for d in self.generator(self.split, self.training_shots):
            self.meta_mlabel_dic[d.meta].add(d.mlabel)
        self.m2idx = {m: i for i, m in enumerate(set(self.mlabels))}
        self.meta2idx = {
            meta: i for i, meta in enumerate(set(self.meta_mlabel_dic.keys()))
        }
        # self.data = [self.get_tensor(d) for d in self.generator(self.split)]
        self.data = [d for d in self.generator(self.split, self.training_shots)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index]

    def collate_fn(self, data):
        data = [self.get_tensor(d) for d in data]
        # print(data[0]["cinput_ids"].size())
        # print(data[1]["cinput_ids"].size())
        data = default_collate(data)
        return data

    def get_tensor(self, ch: CH_Label) -> Dict[str, Any]:
        c, h, multi_c, m, meta = ch

        c_bert, m_c = sentence2bert(c, self.tokenizer, self.length)
        h_bert, m_h = sentence2bert(h, self.tokenizer, self.length)
        multi_c_bert, m_multi_c = sentence2bert(multi_c, self.tokenizer, self.length)

        theta = 0.05
        if random.random() < theta:
            ch_bert, m_ch = sentpair2bert(
                Sentence(tokens=["null"]), h, self.tokenizer, self.length
            )
        else:

            ch_bert, m_ch = sentpair2bert(c, h, self.tokenizer, self.length)

        # nullh_bert, m_nullh = sentpair2bert(Sentence(["NULL"]), h, self.tokenizer, self.length)
        # nullh_dic = {("nullh"+k): torch.squeeze(v,0) for k, v in nullh_bert.items()}

        c_dic = {("c" + k): torch.squeeze(v, 0) for k, v in c_bert.items()}
        h_dic = {("h" + k): torch.squeeze(v, 0) for k, v in h_bert.items()}
        multi_c_dic = {
            ("mulit_c" + k): torch.squeeze(v, 0) for k, v in multi_c_bert.items()
        }
        ch_dic = {("ch" + k): torch.squeeze(v, 0) for k, v in ch_bert.items()}

        marker_dic = {"m_c": m_c, "m_h": m_h, "m_ch": m_ch, "m_multi_c": m_multi_c}
        mlabel_dic = {"mlabel": torch.tensor(self.m2idx[m])}
        meta_dic = {"meta": torch.tensor(self.meta2idx[meta])}
        # meta_dic = {"meta": torch.tensor(nullh_bert)}
        return {
            **meta_dic,
            **mlabel_dic,
            **c_dic,
            **h_dic,
            **ch_dic,
            **marker_dic,
            **multi_c_dic,
            # **nullh_dic
        }


class RandSubClassSampler(torch.utils.data.sampler.Sampler):
    r"""Samples a subset of classes for each mini-batch, without replacement.
    https://github.com/GT-RIPL/L2C/blob/master/dataloaders/sampler.py
    Arguments:
        inds (list): a list of indices
        labels (list): a list of class labels
        cls_per_batch (int): number of class in each mini-batch
        batch_size (int): mini-batch size
        num_batch (int): number of mini-batch
    """

    def __init__(
        self,
        inds,
        labels,
        cls_per_batch,
        batch_size,
        num_batch,
        meta_label_dic,
        meta_per_batch,
    ):
        assert len(inds) == len(labels), "Mismatched inputs inds:%d,labels:%d" % (
            len(inds),
            len(labels),
        )
        self.batch_size = batch_size
        self.cls_per_batch = cls_per_batch
        self.meta_per_batch = meta_per_batch
        self.num_batch = num_batch
        self.sample_per_cls = batch_size // cls_per_batch
        self.inds = inds

        self.labels = labels
        self.cls2ind = {}
        self.label_set = set(labels)
        self.meta_label_dic = meta_label_dic
        for l in self.label_set:
            self.cls2ind[l] = []
        for i in range(len(inds)):
            self.cls2ind[labels[i]].append(inds[i])

        label_cnt = Counter(labels)
        # print(label_cnt)
        self.longtail_labels = set(
            [k for k, v in label_cnt.items() if v < self.sample_per_cls]
        )

    def __iter__(self):
        for b in range(self.num_batch):
            rand_meta_set = random.sample(
                self.meta_label_dic.keys(),
                min(self.meta_per_batch, len(self.meta_label_dic.keys())),
            )
            constrainted_cls_set = [
                label
                for label in chain(
                    *[self.meta_label_dic[meta] for meta in rand_meta_set]
                )
                if label not in self.longtail_labels
            ]
            # print(self.longtail_labels)
            # print(self.meta_label_dic)
            # print(rand_meta_set)
            # print(constrainted_cls_set)
            rand_cls_set = random.sample(
                constrainted_cls_set, min(self.cls_per_batch, len(constrainted_cls_set))
            )
            # print(self.meta_label_dic)
            # print(rand_meta_set)
            # print(constrainted_cls_set)
            # print(self.cls_per_batch)
            # print(rand_cls_set)
            for c in rand_cls_set:
                # assert len(self.cls2ind[c]) >= self.sample_per_cls
                try:
                    sample_per_cls = self.batch_size // len(rand_cls_set)
                    ind_list = random.sample(self.cls2ind[c], sample_per_cls)
                except:
                    print(c)

                    exit()

                for i in ind_list:
                    yield i

    def __len__(self):
        return self.batch_size * self.num_batch


# class Iter_dataset(IterableDataset):
#     def __init__(self, gen, split, weight: Optional[int] = None):
#         self.generator = gen
#         self.weight = weight
#         self.split = split

#     def get_stream(self) -> Generator[QCH, None, None]:
#         for qch in self.generator(self.split):
#             yield qch

#     def __iter__(self) -> Iterator[QCH]:
#         return self.get_stream()


def dataset2loader(
    dataset,
    batch_sz,
    use_sampler=True,
    cls_per_batch=20,
    collate_fn=None,
    meta_per_batch=1,
):
    length = len(dataset)

    # Randomly select 20 characters from 964. By default setting (batch_sz=100), each character has 5 images in a mini-batch.
    if use_sampler:
        idx2mlabel = dataset.mlabels  # train_dataset[i] returns (img, cid)
        sampler = RandSubClassSampler(
            inds=range(length),
            labels=idx2mlabel,
            cls_per_batch=cls_per_batch,
            batch_size=batch_sz,
            num_batch=length // batch_sz,
            meta_label_dic=dataset.meta_mlabel_dic,
            meta_per_batch=meta_per_batch,
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_sz,
            shuffle=False,
            collate_fn=collate_fn,
            # num_workers=num_workers,
            sampler=sampler,
        )
    else:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_sz,
            shuffle=False,
            collate_fn=collate_fn,
            # num_workers=num_workers,
        )
    return loader
