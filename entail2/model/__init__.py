from entail2.model.network import *
from entail2.model.entail2 import *
from entail2.model.crossfit import *
from entail2.model.efl import *
from entail2.model.efl_no_cl import *
from entail2.model.unifew import *
from entail2.model.efl_multichoice import *
from collections import namedtuple


ModelClass = namedtuple("ModelClass", ("tokenizer", "encoder"))


_MODEL_MAP = {
    "entail2": Entail2_contrastive,
    "efl_no_cl": EFL_no_cl,
    "efl": EFL,
    "efl_multichoice": EFL_multichoice,
    "crossfit": Crossfit,
    "unifew": Unifew,
}
_ENCODER_MAP = {
    "bert": ModelClass(
        **{
            "encoder": MyBert("bert-base-uncased"),
            "tokenizer": AutoTokenizer.from_pretrained("bert-base-uncased"),
        }
    ),
    "bart": ModelClass(
        **{
            "encoder": MyBart("facebook/bart-base"),
            "tokenizer": AutoTokenizer.from_pretrained(
                "facebook/bart-base", add_prefix_space=True, use_fast=False
            ),
        }
    ),
}


def get_model(model_name: str):
    return _MODEL_MAP[model_name]


def get_encoder(bert_name: str):
    return _ENCODER_MAP[bert_name]
