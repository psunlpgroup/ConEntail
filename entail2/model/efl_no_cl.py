import torch
from torch import nn

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from entail2.model.network import *


class EFL_no_cl(Model):
    def __init__(self, encoder, tokenizer):
        super().__init__(encoder=encoder, tokenizer=tokenizer)
        self.loss_fct = BCEWithLogitsLoss()
        self.classifier = nn.Linear(768, 1)

    def forward(
        self,
        qinput_ids=None,
        qattention_mask=None,
        cinput_ids=None,
        cattention_mask=None,
        hinput_ids=None,
        hattention_mask=None,
        mulit_cinput_ids=None,
        mulit_cattention_mask=None,
        chinput_ids=None,
        chattention_mask=None,
        m_q=None,
        m_c=None,
        m_h=None,
        m_multi_c=None,
        m_ch=None,
        chtoken_type_ids=None,
        mlabel=None,
        meta=None,
    ):

        # ch = self.encoder(input_ids=chinput_ids, attention_mask=chattention_mask)
        c = self.encoder(input_ids=cinput_ids, attention_mask=cattention_mask)
        logits = self.classifier(c)
        sim = torch.sigmoid(logits)
        # h = self.encoder(input_ids=hinput_ids, attention_mask=hattention_mask)
        mlabel = mlabel.unsqueeze(-1).float()

        # print(mlabel)
        # print(logits)
        # exit()
        # print(mlabel)
        loss = self.loss_fct(logits, mlabel)

        return {"sim": sim, "loss": loss, "blabel": mlabel}

    def predict(
        self,
        qinput_ids=None,
        qattention_mask=None,
        qhinput_ids=None,
        qhattention_mask=None,
        cinput_ids=None,
        cattention_mask=None,
        hinput_ids=None,
        hattention_mask=None,
        mulit_qinput_ids=None,
        mulit_qattention_mask=None,
        chinput_ids=None,
        chattention_mask=None,
        m_q=None,
        m_qh=None,
        m_c=None,
        m_h=None,
        m_multi_q=None,
        m_ch=None,
        chtoken_type_ids=None,
        mlabel=None,
        meta=None,
    ):
        batch_size, support_len, _ = qinput_ids.size()

        # qinput_ids = qinput_ids.view(batch_size * support_len, -1)
        # qattention_mask = qattention_mask.view(batch_size * support_len, -1)

        # view_hinput_ids = hinput_ids.view(batch_size * support_len, -1)
        # hattention_mask = hattention_mask.view(batch_size * support_len, -1)

        qhinput_ids = qhinput_ids.view(batch_size * support_len, -1)
        qhattention_mask = qhattention_mask.view(batch_size * support_len, -1)

        # print(qhinput_ids.size())
        # print(batch_size, support_len)
        # # torch.Size([960, 128])
        # # 32 30

        # exit()
        qh = self.encoder(input_ids=qhinput_ids, attention_mask=qhattention_mask)

        # h = self.encoder(input_ids=view_hinput_ids, attention_mask=hattention_mask)
        qh = self.classifier(qh)
        sim = torch.sigmoid(qh)
        sim = sim.view(-1, support_len)

        max_sim, y_pred_idx = sim.max(dim=1)

        _, true_index = mlabel.view(-1, support_len).max(dim=1)
        y_base_idx = torch.randint_like(y_pred_idx, support_len)

        y_true = self.decode_by_idx(hinput_ids, true_index)
        y_pred = self.decode_by_idx(hinput_ids, y_pred_idx)
        y_base = self.decode_by_idx(hinput_ids, y_base_idx)

        return {
            "sim": sim,
            "mlabel": mlabel,
            "y_pred": y_pred,
            "y_true": y_true,
            "y_base": y_base,
        }
