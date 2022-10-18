import numpy as np
from typing import Dict, List, Tuple, NamedTuple, Optional, Any
from random import sample

from torch import Tensor
import torch
from entail2.common.datastructure import *


def simple_tokenize(text: str) -> List[str]:
    return text.split(" ")


# def pretokenized2bert(toks: List[str], tokenizer):
#     subtoks = []
#     for tok in toks:
#         subtoks.extend(tokenizer.tokenize(tok))
#     return subtoks


def pretokenized2bert(toks: List[str], tokenizer):
    return toks


def sentence2bert(s: Sentence, tokenizer, length):
    bert_dic = {"<unk>": "[UNK]", "<blank_token>": "[MASK]"}
    tokens = pretokenized2bert(s.tokens, tokenizer)
    terms = pretokenized2bert(s.terms, tokenizer)
    tok_bert = [
        "[UNK]" if t == "<unk>" else "[MASK]" if t == "<blank_token>" else t
        for t in tokens
    ]
    # print(length)
    # print(tok_bert)
    term_bert = (
        [False]
        + [True if t in terms else False for t in tok_bert][: (length - 2)]
        + [False]
        + [False] * (length - len(tok_bert) - 2)
    )
    # assert len(term_bert) == length
    tok_bert = tokenizer.encode_plus(
        tok_bert,
        padding="max_length",
        truncation=True,
        max_length=length,
        is_split_into_words=True,
        return_token_type_ids=False,
        return_tensors="pt",
    )
    term_bert = torch.tensor(term_bert)

    # # DEBUG
    # assert tok_bert.input_ids.squeeze().size(0) == 128
    # try:
    #     assert term_bert.squeeze().size(0) == 128
    # except:
    #     print(term_bert)
    #     print(term_bert.size())
    #     exit()

    return tok_bert, term_bert


def sentpair2bert(s_a: Sentence, s_b: Sentence, tokenizer, length):
    bert_dic = {"<unk>": "[UNK]", "<blank_token>": "[MASK]"}
    tokens_a = pretokenized2bert(s_a.tokens, tokenizer)
    tokens_b = pretokenized2bert(s_b.tokens, tokenizer)
    term_a = pretokenized2bert(s_a.terms, tokenizer)
    term_b = pretokenized2bert(s_b.terms, tokenizer)

    tokens_a = [
        "[UNK]" if t == "<unk>" else "[MASK]" if t == "<blank_token>" else t
        for t in tokens_a
    ]
    tokens_b = [
        "[UNK]" if t == "<unk>" else "[MASK]" if t == "<blank_token>" else t
        for t in tokens_b
    ]

    term_a = [False] + [True if t in term_a else False for t in tokens_a]
    term_b = [True if t in term_a else False for t in tokens_b]
    # term_ab = (
    #     [False]
    #     + term_a
    #     + [False]
    #     + term_b
    #     + [False] * (length - len(tokens_b) - len(tokens_a) - 2)
    # )
    # assert len(term_bert) == length
    tok_bert = tokenizer.encode_plus(
        tokens_a,
        tokens_b,
        padding="max_length",
        truncation=True,
        max_length=length,
        is_split_into_words=True,
        return_token_type_ids=True,
        return_tensors="pt",
    )

    # truncate terms according to 'token_type_ids'
    type_mask = (tok_bert["token_type_ids"]).tolist()
    term_a = [t for t, m in zip(term_a, type_mask) if m == 0]

    term_b = [t for t, m in zip(term_b, filter(lambda x: x == 1, type_mask)) if m == 1]
    term_ab = (
        term_a
        + [False]
        + term_b
        + [False]
        + [False] * (length - len(term_b) - len(term_b) - 2)
    )
    term_bert = torch.tensor(term_ab)
    # # DEBUG
    # assert tok_bert.input_ids.squeeze().size(0) == 128
    # assert term_bert.squeeze().size(0) == 128

    return tok_bert, term_bert
