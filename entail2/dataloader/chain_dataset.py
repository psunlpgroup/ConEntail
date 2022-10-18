import torch
from torch.utils.data import IterableDataset, DataLoader
from torch import Tensor
from typing import Dict, Generator, Iterator, List, Optional, Tuple, Any
from random import choices
from pprint import pprint
from itertools import cycle

# from entail2.dataloader import Base_dataset, WikitextDataset, WordnetDataset
from entail2.common import QCH, Sentence, sentence2bert, sentpair2bert

from transformers import DistilBertTokenizerFast


class Chain_dataset(IterableDataset):
    def __init__(self, dataset_list: List[Any], tokenizer) -> None:
        super().__init__()
        self.datasets = dataset_list
        self.len_datasets = len(dataset_list)
        self.wei_datasets = [d.weight for d in dataset_list]
        self.tokenizer = tokenizer
        self.length = 128

    def get_steam(self) -> Generator[Any, None, None]:
        dataset_idx = sum([[i] * w for i, w in enumerate(self.wei_datasets)], [])
        for idx in cycle(dataset_idx):
            for i, item in enumerate(self.datasets[idx]):
                yield item
                if i == 4:
                    break

    def __iter__(self) -> Iterator[Dict[str, Tensor]]:
        yield from map(self.get_tensor, self.get_steam())

    def get_tensor(self, item: Any) -> Dict[str, Tensor]:
        raise NotImplementedError


class Chain_QCH_dataset(Chain_dataset):
    def get_tensor(self, qch: QCH) -> Dict[str, Tensor]:
        q, c, h, b, _, _ = qch

        q, m_q = sentence2bert(q, self.tokenizer, self.length)
        ch, m_ch = sentpair2bert(c, h, self.tokenizer, self.length)

        q_dic = {("q" + k): torch.squeeze(v, 0) for k, v in q.items()}
        ch_dic = {("ch" + k): torch.squeeze(v, 0) for k, v in ch.items()}

        bdic = {"b": torch.tensor(b).long()}
        marker_dic = {"m_q": m_q, "m_ch": m_ch}
        return {**bdic, **q_dic, **ch_dic, **marker_dic}

    # def get_tensor(self, qch: QCH) -> Dict[str, Tensor]:
    #     q, c, h, b, _, _ = qch

    #     q, mq = sentence2bert(q, self.tokenizer, self.length)
    #     c, mc = sentence2bert(c, self.tokenizer, self.length)
    #     h, mh = sentence2bert(h, self.tokenizer, self.length)

    #     qdic = {("q" + k): v for k, v in q.items()}
    #     cdic = {("c" + k): v for k, v in c.items()}
    #     hdic = {("h" + k): v for k, v in h.items()}

    #     bdic = {"b": torch.tensor(b)}
    #     marker_dic = {"mq": mq, "mc": mc, "mh": mh}
    #     return {**bdic, **qdic, **cdic, **hdic, **marker_dic}


class Chain_dataloader:
    def __init__(
        self,
        loader_funs,
        batch_size: int,
        split: str,
        tokenizer,
        use_sampler: bool,
        training_shots: int,
    ) -> None:

        assert split in ("train", "eval")
        if split == "train":
            self.loaders = [
                f(batch_size, tokenizer, use_sampler, training_shots)[0]
                for f in loader_funs
                if f(batch_size, tokenizer, use_sampler, training_shots)[0] is not None
            ]
        else:
            self.loaders = list(
                filter(
                    lambda x: len(x) > 0,
                    [
                        f(batch_size, tokenizer)[1]
                        for f in loader_funs
                        if f(batch_size, tokenizer)[1] is not None
                    ],
                )
            )

        self.weights = [1] * len(self.loaders)

    def __iter__(self):
        for i, w in enumerate(self.weights):
            for t in range(w):
                for batch in self.loaders[i]:
                    yield batch

    def __len__(self):
        return sum(map(len, self.loaders))


if __name__ == "__main__":
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    pass
