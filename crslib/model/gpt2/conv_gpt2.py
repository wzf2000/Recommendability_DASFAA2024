import os

import torch
from torch import nn
from transformers import GPT2LMHeadModel

from crslab.config import PRETRAIN_PATH
from crslab.model.base import BaseModel
from crslab.model.pretrained_models import resources

from crslib.loss import BinaryFocalLoss, GMHCLoss, BinaryDSCLoss


class ConvGPT2Model(BaseModel):

    def __init__(self, opt, device, side_data):
        """

        Args:
            opt (dict): A dictionary record the hyper parameters.
            device (torch.device): A variable indicating which device to place the data and model.
            side_data (dict): A dictionary record the side data.
        
        """
        language = opt['language']
        assert opt['tokenize'] == 'gpt2'
        assert language in ['en', 'zh']
        super(BaseModel, self).__init__()
        self.opt = opt
        self.device = device
        self.build_model()

    def build_model(self, *args, **kwargs):
        """build model"""
        if self.opt['language'] == 'en':
            self.context_gpt2: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained('gpt2')
        else:
            self.context_gpt2: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained('IDEA-CCNL/Wenzhong-GPT2-110M')

        self.gpt2_hidden_size = self.context_gpt2.config.n_embd
        self.state2score = nn.Linear(self.gpt2_hidden_size, 1)

        if self.opt['loss'] == 'BCEWithLogitsLoss':
            if 'pos_weight' not in self.opt:
                self.opt['pos_weight'] = 1.0
            self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.opt['pos_weight']]))
        elif self.opt['loss'] == 'BinaryFocalLoss':
            if 'gamma' not in self.opt:
                self.opt['gamma'] = 2.0
            if 'alpha' not in self.opt:
                self.opt['alpha'] = 0.5
            self.loss = BinaryFocalLoss(gamma=self.opt['gamma'], alpha=self.opt['alpha'])
        elif self.opt['loss'] == 'GMHCLoss':
            if 'bins' not in self.opt:
                self.opt['bins'] = 10
            if 'alpha' not in self.opt:
                self.opt['alpha'] = 0.5
            self.loss = GMHCLoss(bins=self.opt['bins'], alpha=self.opt['alpha'])
        elif self.opt['loss'] == 'BinaryDSCLoss':
            if 'alpha' not in self.opt:
                self.opt['alpha'] = 1.0
            if 'smooth' not in self.opt:
                self.opt['smooth'] = 1.0
            self.loss = BinaryDSCLoss(alpha=self.opt['alpha'], smooth=self.opt['smooth'])
        else:
            raise NotImplementedError

    def forward(self, batch, mode):
        # conv_id, message_id, context, context_mask, topic_path_kw, tp_mask, user_profile, profile_mask, y = batch
        context, context_mask, goals, goal_mask, user_profile, profile_mask, y = batch

        context_rep = self.context_gpt2(
            context,
            attention_mask=context_mask,
            output_hidden_states=True,
        ).hidden_states[0][:, 0]  # [bs, hidden_size]

        scores = self.state2score(context_rep)[:, 0]

        loss = self.loss(scores, y)

        return loss, torch.sigmoid(scores)
