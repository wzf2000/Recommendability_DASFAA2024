import os

import torch
from torch import nn
from transformers import DebertaModel, AutoModel
from transformers.models.bert.modeling_bert import BertPooler

from crslab.model.base import BaseModel

from crslib.loss import BinaryFocalLoss


class ConvDeBERTaModel(BaseModel):

    def __init__(self, opt, device, side_data):
        """

        Args:
            opt (dict): A dictionary record the hyper parameters.
            device (torch.device): A variable indicating which device to place the data and model.
            side_data (dict): A dictionary record the side data.
        
        """
        assert opt['tokenize'] == 'deberta'
        super(BaseModel, self).__init__()
        self.opt = opt
        self.device = device
        self.build_model()

    def build_model(self, *args, **kwargs):
        """build model"""
        if self.opt['language'] == 'en':
            self.context_deberta: DebertaModel = DebertaModel.from_pretrained('microsoft/deberta-base')
        elif self.opt['language'] == 'zh':
            self.context_deberta: DebertaModel = AutoModel.from_pretrained('IDEA-CCNL/Erlangshen-DeBERTa-v2-186M-Chinese-SentencePiece')
        else:
            raise NotImplementedError

        self.deberta_hidden_size = self.context_deberta.config.hidden_size
        self.pooler = BertPooler(self.context_deberta.config)
        self.state2score = nn.Linear(self.deberta_hidden_size, 1)

        if self.opt['loss'] == 'BCEWithLogitsLoss':
            if 'pos_weight' not in self.opt:
                self.opt['pos_weight'] = 1.0
            self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.opt['pos_weight']]).to(self.device))
        elif self.opt['loss'] == 'BinaryFocalLoss':
            if 'gamma' not in self.opt:
                self.opt['gamma'] = 2.0
            if 'alpha' not in self.opt:
                self.opt['alpha'] = 0.5
            self.loss = BinaryFocalLoss(gamma=self.opt['gamma'], alpha=self.opt['alpha'])
        else:
            raise NotImplementedError

    def forward(self, batch, mode):
        # conv_id, message_id, context, context_mask, topic_path_kw, tp_mask, user_profile, profile_mask, y = batch
        context, context_mask, goals, goal_mask, user_profile, profile_mask, y = batch

        context_rep = self.context_deberta(
            context,
            context_mask,
        ).last_hidden_state  # [bs, seq_len, hidden_size]
        context_rep = self.pooler(context_rep)  # [bs, hidden_size]

        scores = self.state2score(context_rep)[:, 0]

        loss = self.loss(scores, y)

        return loss, torch.sigmoid(scores)
