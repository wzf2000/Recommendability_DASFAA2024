import os

import torch
from torch import nn
from transformers import BertModel

from crslab.config import PRETRAIN_PATH
from crslab.data import dataset_language_map
from crslab.model.base import BaseModel
from crslab.model.pretrained_models import resources


class ProfileBERTModel(BaseModel):

    def __init__(self, opt, device, vocab, side_data):
        """

        Args:
            opt (dict): A dictionary record the hyper parameters.
            device (torch.device): A variable indicating which device to place the data and model.
            vocab (dict): A dictionary record the vocabulary information.
            side_data (dict): A dictionary record the side data.
        
        """
        language = opt['language']
        resource = resources['bert'][language]
        dpath = os.path.join(PRETRAIN_PATH, "bert", language)
        super().__init__(opt, device, dpath, resource)

    def build_model(self, *args, **kwargs):
        """build model"""
        self.profile_bert = BertModel.from_pretrained(self.dpath)

        self.bert_hidden_size = self.profile_bert.config.hidden_size
        self.state2score = nn.Linear(self.bert_hidden_size, 1)

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, batch, mode):
        # conv_id, message_id, context, context_mask, topic_path_kw, tp_mask, user_profile, profile_mask, y = batch
        context, context_mask, goals, goal_mask, user_profile, profile_mask, y = batch

        profile_rep = self.profile_bert(
            user_profile,
            profile_mask,
        ).pooler_output  # [bs, hidden_size]

        scores = self.state2score(profile_rep)[:, 0]

        loss = self.loss(scores, y)

        return loss, torch.sigmoid(scores)
