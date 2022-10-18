import torch
import torch.nn.functional as F
from torch import Tensor, nn


from entail2.model.network import *


def label_smoothed_nll_loss(lprobs, target, epsilon=0.1, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


class Unifew(Model):
    def __init__(self, encoder, tokenizer):
        super().__init__(encoder=encoder, tokenizer=tokenizer)

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
        loss = None
        outputs = self.encoder(
            mulit_cinput_ids,
            cattention_mask,
            hinput_ids=hinput_ids,
            hattention_mask=hattention_mask,
            # is_training=is_training,
        )

        lm_logits = F.linear(
            outputs[0],
            self.encoder.bart.model.shared.weight,
            bias=self.encoder.bart.final_logits_bias,
        )
        lprobs = F.log_softmax(lm_logits, dim=-1)
        loss, _ = label_smoothed_nll_loss(
            lprobs,
            hinput_ids,
            epsilon=0.1,
            ignore_index=self.encoder.bart.config.pad_token_id,
        )
        # return loss
        return {"loss": loss, "lm_logits": lm_logits, "outputs": outputs[1:]}

        # return (lm_logits,) + outputs[1:]

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
        # batch[0], batch[1] = trim_batch(batch[0], pad_token_id, batch[1])
        predictions = []
        y_true = []
        batch_size, support_len, _ = qinput_ids.size()

        # qinput_ids = qinput_ids[:, 0, :]
        # qattention_mask = qattention_mask[:, 0, :]

        mulit_qinput_ids = mulit_qinput_ids[:, 0, :]
        mulit_qattention_mask = mulit_qattention_mask[:, 0, :]

        outputs = self.encoder.bart.generate(
            input_ids=mulit_qinput_ids,
            attention_mask=mulit_qattention_mask,
            num_beams=4,
            max_length=64,
            decoder_start_token_id=self.encoder.bart.model.config.bos_token_id,
            early_stopping=False,
        )
        for input_, output in zip(qinput_ids, outputs):
            pred = self.tokenizer.decode(
                output, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            predictions.append(pred)
        _, true_index = mlabel.view(-1, support_len).max(dim=1)

        y_true = self.decode_by_idx(hinput_ids, true_index)
        y_pred = predictions
        y_base = ["nan" for i in y_true]

        return {
            "mlabel": mlabel,
            "y_pred": y_pred,
            "y_true": y_true,
            "y_base": y_base,
        }
