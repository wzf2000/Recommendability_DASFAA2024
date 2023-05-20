import os

import torch
from torch import nn
from transformers import BertModel, RobertaModel

from crslab.config import PRETRAIN_PATH
from crslab.data import dataset_language_map
from crslab.model.base import BaseModel
from crslab.model.pretrained_models import resources

from crslib.loss import BinaryFocalLoss, GMHCLoss, BinaryDSCLoss


class ConvBERTModel(BaseModel):

    def __init__(self, opt, device, vocab, side_data):
        """

        Args:
            opt (dict): A dictionary record the hyper parameters.
            device (torch.device): A variable indicating which device to place the data and model.
            vocab (dict): A dictionary record the vocabulary information.
            side_data (dict): A dictionary record the side data.
        
        """
        language = opt['language']
        assert language in ['en', 'zh']
        assert opt['tokenize'] in ['bert', 'roberta']
        if opt['tokenize'] == 'bert':
            resource = resources[opt['tokenize']][language]
            dpath = os.path.join(PRETRAIN_PATH, opt['tokenize'], language)
            super().__init__(opt, device, dpath, resource)
        elif opt['tokenize'] == 'roberta':
            super(BaseModel, self).__init__()
            self.opt = opt
            self.device = device
            self.build_model()
        else:
            raise NotImplementedError

    def build_model(self, *args, **kwargs):
        """build model"""
        if self.opt['tokenize'] == 'bert':
            self.context_bert: BertModel = BertModel.from_pretrained(self.dpath)
        else:
            if self.opt['language'] == 'en':
                self.context_bert: RobertaModel = RobertaModel.from_pretrained('roberta-base')
            elif self.opt['language'] == 'zh':
                # self.context_bert: BertModel = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
                self.context_bert: BertModel = BertModel.from_pretrained('IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment')
            else:
                raise NotImplementedError

        self.bert_hidden_size = self.context_bert.config.hidden_size
        self.state2score = nn.Linear(self.bert_hidden_size, 1)

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

        context_rep = self.context_bert(
            context,
            context_mask,
        ).pooler_output  # [bs, hidden_size]

        scores = self.state2score(context_rep)[:, 0]

        loss = self.loss(scores, y)

        return loss, torch.sigmoid(scores)
