from typing import Dict, List, Tuple, NamedTuple, Optional, Any
from random import sample

from torch import Tensor
import torch

# from collections import namedtuple

# Q C H: tokens, terms
class Sentence(NamedTuple):
    tokens: List[str]
    terms: List[str] = []
    meta: Any = None

    def __add__(self, s):
        return Sentence(
            tokens=self.tokens + ["[SEP]"] + s.tokens,
            terms=self.terms + s.terms,
            meta=None,
        )


class QCH(NamedTuple):
    query: Sentence
    context: Sentence
    hypothesis: Sentence
    multichoice_query: Sentence
    query_hypo: Optional[Sentence] = None
    blabel: Optional[bool] = None  # binary label, True or False
    mlabel: Optional[str] = None  # multi label
    meta: Any = None


class CH_Label(NamedTuple):
    context: Sentence
    hypothesis: Sentence
    multichoice_context: Sentence
    mlabel: Optional[str] = None
    meta: Any = None


# def qch2tensor(qch: QCH, tokenizer, length) -> Dict[str, Tensor]:
#     q, c, h, b, _, _ = qch

#     q, m_q = sentence2bert(q, tokenizer, length)
#     ch, m_ch = sentpair2bert(c, h, tokenizer, length)

#     q_dic = {("q" + k): torch.squeeze(v, 0) for k, v in q.items()}
#     ch_dic = {("ch" + k): torch.squeeze(v, 0) for k, v in ch.items()}

#     bdic = {"b": torch.tensor(b).long()}
#     marker_dic = {"m_q": m_q, "m_ch": m_ch}
#     return {**bdic, **q_dic, **ch_dic, **marker_dic}


def CH_sampler(QCL_d: Dict[str, List[CH_Label]]):
    # weights = list(map(len, QCL_d.values()))
    a, b = sample(list(QCL_d.values()), k=2)
    a1, a2 = sample(a, k=2)
    b1, b2 = sample(b, k=2)
    CH1 = CH(context=a1.context, hypothesis=a1.hypothesis, mlabel=a1.mlabel)
    CH2 = CH(context=a2.context, hypothesis=a2.hypothesis, mlabel=a2.mlabel)
    CH3 = CH(context=b1.context, hypothesis=b1.hypothesis, mlabel=b1.mlabel)
    CH4 = CH(context=b2.context, hypothesis=b2.hypothesis, mlabel=b2.mlabel)
    for ch in [CH1, CH2, CH3, CH4]:
        yield ch


def QCH_sampler(QCL_d: Dict[str, List[CH_Label]]):
    # weights = list(map(len, QCL_d.values()))
    a, b = sample(list(QCL_d.values()), k=2)
    a1, a2 = sample(a, k=2)
    b1, b2 = sample(b, k=2)

    # QCH = Q_a1 C_b1 H_b1 = False
    # QCH = Q_b1 C_a1 H_a1 = False
    # QCH = Q_a1 C_a2 H_a2 = True
    # QCH = Q_b1 C_b2 H_b2 = True
    QCH1 = QCH(
        query=a1.context,
        context=b1.context,
        hypothesis=b1.hypothesis,
        blabel=False,
        mlabel=b1.mlabel,
    )

    QCH2 = QCH(
        query=b1.context,
        context=a1.context,
        hypothesis=a1.hypothesis,
        blabel=False,
        mlabel=a1.mlabel,
    )

    QCH3 = QCH(
        query=a1.context,
        context=a2.context,
        hypothesis=a2.hypothesis,
        blabel=True,
        mlabel=a1.mlabel,
    )

    QCH4 = QCH(
        query=b1.context,
        context=b2.context,
        hypothesis=b2.hypothesis,
        blabel=True,
        mlabel=b1.mlabel,
    )

    for qch in [QCH1, QCH2, QCH3, QCH4]:
        yield qch
