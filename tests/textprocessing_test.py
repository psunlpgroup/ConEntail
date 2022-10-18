from entail2.common import *
from transformers import DistilBertTokenizerFast
from transformers import AutoTokenizer

import pytest

import sys

sys.path.insert(0, ".")


class TestTextProcessing:
    def test_text2bert(self):
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased")
        wikitext = QCH(
            query=Sentence(
                tokens=[
                    "travelling",
                    "across",
                    "the",
                    "land",
                    "of",
                    "the",
                    "native",
                    "<unk>",
                    "people",
                    "nganno",
                    "was",
                    "wounded",
                    "in",
                    "a",
                    "battle",
                    "and",
                    "laid",
                    "down",
                    "to",
                    "die",
                    "forming",
                    "the",
                    "mount",
                    "<blank_token>",
                    "ranges",
                ],
                terms=["lofty"],
                meta=None,
            ),
            context=Sentence(
                tokens=[
                    "this",
                    "genus",
                    "of",
                    "conifers",
                    "was",
                    "native",
                    "in",
                    "europe",
                    "until",
                    "the",
                    "<unk>",
                    "and",
                    "<unk>",
                    "<unk>",
                    "N",
                    "million",
                    "to",
                    "N",
                    "N",
                    "million",
                    "years",
                    "ago",
                    "but",
                    "became",
                    "extinct",
                    "there",
                    "so",
                    "the",
                    "amylostereum",
                    "fungi",
                    "specialized",
                    "on",
                    "other",
                    "conifers",
                    "and",
                    "<blank_token>",
                    "into",
                    "several",
                    "species",
                ],
                terms=["differentiated"],
                meta=None,
            ),
            hypothesis=Sentence(tokens=["differentiated"], terms=None, meta=None),
            blabel=False,
            mlabel="differentiated",
            meta=None,
        )

        wiki_bert, terms = sentence2bert(wikitext.query, tokenizer, 128)
        assert wiki_bert.input_ids.squeeze().size(0) == 128
        assert wiki_bert.input_ids.squeeze().size() == terms.size()

        truncate_sent = Sentence(tokens=["hello"] * 1000, terms=["emmm"])
        wiki_bert, terms = sentence2bert(truncate_sent, tokenizer, 128)
        assert wiki_bert.input_ids.squeeze().size(0) == 128
        assert wiki_bert.input_ids.squeeze().size() == terms.size()

        truncate_sent = Sentence(tokens=["hello"] * 1000, terms=["emmm"])
        wiki_bert, terms = sentpair2bert(truncate_sent, truncate_sent, tokenizer, 128)
        assert wiki_bert.input_ids.squeeze().size(0) == 128
        assert wiki_bert.input_ids.squeeze().size() == terms.size()

    def test_padding(self):
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased")
        sent = [
            "travelling",
            "across",
            "the",
            "land",
            "of",
            "the",
            "native",
            "<unk>",
            "people",
            "nganno",
        ]
        tok_bert_short = tokenizer.encode_plus(
            sent,
            padding="max_length",
            truncation=True,
            is_split_into_words=True,
            max_length=100,
        )["input_ids"]
        tok_bert_long = tokenizer.encode_plus(
            sent * 50,
            padding="max_length",
            truncation=True,
            is_split_into_words=True,
            max_length=100,
        )["input_ids"]
        assert len(tok_bert_short) == 100
        assert len(tok_bert_long) == 100

    def test_truncation(self):
        from transformers import BertTokenizer

        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        sequence_a = "HuggingFace is based in NYC"
        sequence_b = "Where is HuggingFace based?"

        encoded_dict = tokenizer(sequence_a, sequence_b)
        print(encoded_dict["token_type_ids"])
        print(tokenizer.tokenize(sequence_a))
        print(tokenizer.tokenize(sequence_b))

        decoded = tokenizer.decode(encoded_dict["input_ids"])
        assert encoded_dict["token_type_ids"] == [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ]

    def test_pretokenize(self):
        sent = ["Hello", "I'm", "a", "single", "sentence", "huggingface"]
        subtokens = []
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        encoded_input = tokenizer(sent, is_split_into_words=True)
        out = tokenizer.decode(encoded_input["input_ids"])
        assert out == "[CLS] Hello I'm a single sentence huggingface [SEP]"
        subtokens = pretokenized2bert(sent, tokenizer)
        print(subtokens)
        assert subtokens == [
            "Hello",
            "I",
            "'",
            "m",
            "a",
            "single",
            "sentence",
            "hugging",
            "##face",
        ]
