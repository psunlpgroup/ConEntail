import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from entail2.model.network import *


class Entail2_contrastive(Model):
    def __init__(self, encoder, tokenizer):
        super().__init__(encoder=encoder, tokenizer=tokenizer)
        self.contrastive = Cartesian_similarity()
        self.cos = nn.CosineSimilarity(dim=-1)

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
        # nullhinput_ids=None,
        # nullhattention_mask=None,
        # nullhtoken_type_ids=None,
    ):
        # c = self.encoder(input_ids=cinput_ids, attention_mask=cattention_mask)
        # h_tokens = self.tokenizer.decode(hinput_ids)
        # print(h_tokens)
        # exit()

        # unique_nullh, unique_idx = torch.unique(nullhinput_ids, dim=0, return_inverse=True)
        # unique_idx = unique_idx.tolist()
        # unique_select_idx = [unique_idx.index(i) for i in dict.fromkeys(unique_idx)]
        # unique_nullhattention_mask = nullhattention_mask[unique_select_idx, :]
        # unique_label = mlabel[unique_select_idx]
        # print(unique_idx)
        # print(unique_nullh.size())
        # print(nullhattention_mask.size())
        # print(unique_select_idx)
        # print(unique_nullhattention_mask.size())

        # exit()
        c = self.encoder(
            input_ids=mulit_cinput_ids, attention_mask=mulit_cattention_mask
        )

        # ch_and_unique_null_ids = torch.cat((chinput_ids, unique_nullh), dim=0)
        # ch_and_unique_null_att_mask = torch.cat((chattention_mask, unique_nullhattention_mask), dim=0)

        ch = self.encoder(input_ids=chinput_ids, attention_mask=chattention_mask)
        # ch = self.encoder(input_ids=ch_and_unique_null_ids, attention_mask=ch_and_unique_null_att_mask)
        # cat_label = torch.cat((mlabel, unique_label), dim=0)
        sim, loss, y_true = self.contrastive(c, ch, mlabel, meta)
        return {"sim": sim, "loss": loss, "blabel": y_true}

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

        qinput_ids = mulit_qinput_ids.view(batch_size * support_len, -1)
        qattention_mask = mulit_qattention_mask.view(batch_size * support_len, -1)

        chinput_ids = chinput_ids.view(batch_size * support_len, -1)
        chattention_mask = chattention_mask.view(batch_size * support_len, -1)

        q = self.encoder(input_ids=qinput_ids, attention_mask=qattention_mask)

        ch = self.encoder(input_ids=chinput_ids, attention_mask=chattention_mask)

        sim = self.cos(q, ch)
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

    def topk_predict(
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

        qinput_ids = mulit_qinput_ids.view(batch_size * support_len, -1)
        qattention_mask = mulit_qattention_mask.view(batch_size * support_len, -1)

        chinput_ids = chinput_ids.view(batch_size * support_len, -1)
        chattention_mask = chattention_mask.view(batch_size * support_len, -1)

        q = self.encoder(input_ids=qinput_ids, attention_mask=qattention_mask)

        ch = self.encoder(input_ids=chinput_ids, attention_mask=chattention_mask)

        sim = self.cos(q, ch)
        sim = sim.view(-1, support_len)

        # k = min(support_len, 10)
        k = support_len
        max_sim, y_pred_idx = torch.topk(sim, k, dim=1)
        all_pred_label = []
        # print(y_pred_idx.size())
        # print(hinput_ids.size())
        # print(chinput_ids.size())
        for b in range(y_pred_idx.size(0)):
            ins = []
            for i in range(k):
                pred_b = self.decode_by_idx(hinput_ids, y_pred_idx[b, i])
                print(pred_b)
                # assert pred_b[0] == pred_b[1]
                ins.append(pred_b[0])
            all_pred_label.append(ins)

        # print(max_sim)
        # print(y_pred_idx)
        # exit()
        result = {"top_sim": max_sim.tolist(), "pred_label": all_pred_label}
        # for
        # _, true_index = mlabel.view(-1, support_len).max(dim=1)
        # y_base_idx = torch.randint_like(y_pred_idx, support_len)

        # y_true = self.decode_by_idx(hinput_ids, true_index)
        # y_pred = self.decode_by_idx(hinput_ids, y_pred_idx)
        # y_base = self.decode_by_idx(hinput_ids, y_base_idx)

        return result
