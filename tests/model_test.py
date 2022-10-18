from entail2.common import sentence2bert
from entail2.model import *
import torch

import pytest


class Test_model:
    def test_contrastive(self):
        cos = Cartesian_similarity()
        a = torch.rand((32, 768))
        b = torch.rand((32, 768))

        mlabel = torch.randint(low=0, high=5, size=(32,))
        sim, loss, score = cos(a, b, mlabel)
        assert tuple(score.size()) == (32, 32)
        assert tuple(loss.size()) == ()
        assert loss > 0

    def test_cuda(self):
        model = Entail2_contrastive("distilbert-base-uncased")
        model = model.cuda()
