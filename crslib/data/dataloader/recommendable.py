import torch
from crslab.data.dataloader import BaseDataLoader
from crslab.data.dataloader.utils import add_start_end_token_idx, padded_tensor, truncate, merge_utt
from tqdm import tqdm
from copy import deepcopy

class RecommendableDataLoader(BaseDataLoader):
    def __init__(self, opt, dataset, vocab):
        super().__init__(opt, dataset)
        self.vocab = vocab
        self.pad_token_idx = vocab['pad_token']
        self.cls_id = vocab['cls_token']
        self.start_token_idx = self.cls_id
        self.sep_id = vocab['sep_token']
        self.unk_token_idx = vocab['unk_token']
        self.conv_bos_id = self.cls_id
        self.end_token_idx = self.sep_id
        self.sent_split_idx = self.sep_id

        self.context_truncate = opt.get('context_truncate', None)
        self.response_truncate = opt.get('response_truncate', None)
        self.item_truncate = opt.get('item_truncate', None)

    def get_recommendable_data(self, batch_size, shuffle=True):
        return self.get_data(self.recommendable_batchify, batch_size, shuffle, self.recommendable_process_fn)

    def recommendable_process_fn(self, *args, **kwargs):
        return self.dataset

    def recommendable_batchify(self, batch):
        batch_context = []
        batch_context_goal = []
        batch_user_profile = []
        batch_label = []

        for conv_dict in batch:
            context = conv_dict['context_tokens']
            context = merge_utt(context, None, False, None)
            context = add_start_end_token_idx(
                truncate(context, max_length=self.context_truncate - 1, truncate_tail=False),
                start_token_idx=self.cls_id,
            )
            batch_context.append(context)

            # [goal, goal, ..., goal]
            context_goals = conv_dict['context_goals']
            context_goals = merge_utt(context_goals, None, False, None)
            context_goals = add_start_end_token_idx(
                context_goals,
                start_token_idx=self.cls_id,
            )
            batch_context_goal.append(context_goals)

            user_profile = conv_dict['user_profile']
            user_profile = merge_utt(user_profile, None, False, None)
            user_profile = add_start_end_token_idx(
                truncate(user_profile, max_length=self.context_truncate - 1, truncate_tail=False),
                start_token_idx=self.cls_id,
            )
            batch_user_profile.append(user_profile)

            batch_label.append(conv_dict['label'])

        batch_context = padded_tensor(batch_context,
                                      pad_idx=self.pad_token_idx,
                                      pad_tail=True,
                                      max_len=self.context_truncate)
        batch_cotnext_mask = (batch_context != self.pad_token_idx).long()
        batch_context_goal = padded_tensor(batch_context_goal,
                                           pad_idx=self.pad_token_idx,
                                           pad_tail=True)
        batch_context_policy_mask = (batch_context_goal != self.pad_token_idx).long()
        batch_user_profile = padded_tensor(batch_user_profile,
                                           pad_idx=self.pad_token_idx,
                                           pad_tail=True)
        batch_user_profile_mask = (batch_user_profile != self.pad_token_idx).long()
        batch_label = torch.tensor(batch_label, dtype=torch.float32)

        return (batch_context, batch_cotnext_mask, batch_context_goal,
                batch_context_policy_mask, batch_user_profile,
                batch_user_profile_mask, batch_label)
