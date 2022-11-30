from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import BartForConditionalGeneration

import torch

from abc import ABC, abstractmethod


class Model(nn.Module):
    def __init__(self, encoder, tokenizer):
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer

    @abstractmethod
    def predict(
        self,
        qinput_ids=None,
        qattention_mask=None,
        cinput_ids=None,
        cattention_mask=None,
        hinput_ids=None,
        hattention_mask=None,
        chinput_ids=None,
        chattention_mask=None,
        m_q=None,
        m_c=None,
        m_h=None,
        m_ch=None,
        chtoken_type_ids=None,
        mlabel=None,
        meta=None,
    ):
        NotImplementedError

    def set_encoder(self, encoder):
        self.encoder = encoder

    def decode_by_idx(self, hinput_ids, true_index):
        y_true = []
        y_true_tokenid = hinput_ids[torch.arange(hinput_ids.size(0)), true_index]

        for token_seq in y_true_tokenid:
            y_true_tokens = self.tokenizer.decode(
                token_seq, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            y_true.append(y_true_tokens)
        return y_true


class MyBart(nn.Module):
    def __init__(self, bertname):
        super().__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(bertname)

    def forward(
        self,
        cinput_ids,
        cattention_mask,
        hinput_ids=None,
        hattention_mask=None,
        # is_training=True,
    ):
        if self.training:
            _decoder_input_ids = shift_tokens_right(
                hinput_ids, self.bart.config.pad_token_id
            )
        else:
            _decoder_input_ids = hinput_ids

        outputs = self.bart.model(
            cinput_ids,
            attention_mask=cattention_mask,
            encoder_outputs=None,
            decoder_input_ids=_decoder_input_ids,
            decoder_attention_mask=hattention_mask,
            return_dict=True,
        )
        return outputs


class MyBert(nn.Module):
    def __init__(self, bertname):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bertname)
        self.pooler = Pooler(pooler_type="cls")

    def forward(self, input_ids, attention_mask):
        # TODO marker
        out = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        out = self.pooler(attention_mask=attention_mask, outputs=out)
        return out


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in [
            "cls",
            "cls_before_pooler",
            "avg",
            "avg_top2",
            "avg_first_last",
        ], (
            "unrecognized pooling type %s" % self.pooler_type
        )

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        hidden_states = outputs.hidden_states

        if self.pooler_type in ["cls_before_pooler", "cls"]:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return (last_hidden * attention_mask.unsqueeze(-1)).sum(
                1
            ) / attention_mask.sum(-1).unsqueeze(-1)
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = (
                (first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)
            ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = (
                (last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)
            ).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens


class Cosine_similarity(nn.Module):
    def __init__(self, temp=0.05):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Dot_similarity(nn.Module):
    # TODO: check size
    def __init__(self, temp=0.05):
        super().__init__()
        self.temp = temp

    def forward(self, x, y):
        return torch.div(torch.matmul(x, y.T), self.temp)


class Cartesian_similarity(nn.Module):
    """https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    Add meta-task mask.
    """

    def __init__(self):
        super().__init__()
        self.temp = 0.05
        self.similarity = Cosine_similarity(self.temp)

    def forward(self, c, ch, m_label=None, meta=None):
        sim = self.similarity(c.unsqueeze(1), ch.unsqueeze(0))
        loss = 0
        y_true = 0
        if m_label is not None:
            # mask Entail(q_a, c_a, h_a)
            # if they are the same ch
            y_true = (m_label.unsqueeze(1) == m_label.unsqueeze(0)).long()
            mask = m_label.unsqueeze(1) == m_label.unsqueeze(0)

            # if meta is not None:
            #     # mask pairs not in the same task
            #     # if they are NOT in the same task
            #     meta_mask = meta.unsqueeze(1) != meta.unsqueeze(0)
            #     mask = mask | meta_mask

            mask = mask.float()
            # # compute cos logits
            # sim = torch.div(
            #     torch.matmul(anchor_feature, contrast_feature.T), self.temperature
            # )

            # for numerical stability
            logits_max, _ = torch.max(sim, dim=1, keepdim=True)
            logits = sim - logits_max.detach()

            # tile mask
            # mask = mask.repeat(anchor_count, contrast_count)
            # mask-out self-contrast cases
            logits_mask = 1
            mask = mask * logits_mask

            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

            # loss
            loss = -mean_log_prob_pos.mean()
        return sim * self.temp, loss, y_true


class Deprecated_Cartesian_similarity(nn.Module):
    """https://github.com/HobbitLong/SupContrast/blob/master/losses.py"""

    def __init__(self):
        super().__init__()
        self.temp = 0.05
        self.similarity = Cosine_similarity(self.temp)

    def forward(self, c, ch, m_label=None):
        sim = self.similarity(c.unsqueeze(1), ch.unsqueeze(0))
        loss = 0
        y_true = 0
        if m_label is not None:
            y_true = (m_label.unsqueeze(1) == m_label.unsqueeze(0)).long()
            mask = (m_label.unsqueeze(1) == m_label.unsqueeze(0)).float()
            # # compute cos logits
            # sim = torch.div(
            #     torch.matmul(anchor_feature, contrast_feature.T), self.temperature
            # )

            # for numerical stability
            logits_max, _ = torch.max(sim, dim=1, keepdim=True)
            logits = sim - logits_max.detach()

            # tile mask
            # mask = mask.repeat(anchor_count, contrast_count)
            # mask-out self-contrast cases
            logits_mask = 1
            mask = mask * logits_mask

            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

            # loss
            loss = -mean_log_prob_pos.mean()
        return sim * self.temp, loss, y_true
