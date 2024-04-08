import torch
from crslab.data.dataloader import BaseDataLoader
from crslab.data.dataloader.utils import add_start_end_token_idx, padded_tensor, truncate, merge_utt
from transformers import AutoTokenizer, BertTokenizer, GPT2Tokenizer, DebertaTokenizer, RobertaTokenizer, PreTrainedTokenizer
from tqdm import tqdm
from copy import deepcopy
from loguru import logger

class RecommendableDataLoader(BaseDataLoader):
    def __init__(self, opt, dataset):
        super().__init__(opt, dataset)

        self.opt = opt
        self.language = opt['language']
        self.tokenize = opt['tokenize']
        self.get_tokenizer()
        self.pad_token_idx = self.tokenizer.pad_token_id

        self.context_truncate = opt.get('context_truncate', 512)
        self.response_truncate = opt.get('response_truncate', 30)
        self.entity_truncate = opt.get('entity_truncate', 30)
        self.word_truncate = opt.get('word_truncate', 30)
        self.item_truncate = opt.get('item_truncate', 30)
        logger.info(f'[Dataset length is {len(self.dataset)}]')

    def get_tokenizer(self) -> PreTrainedTokenizer:
        if self.tokenize == 'bert':
            if self.language == 'zh':
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            elif self.language == 'en':
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            else:
                raise NotImplementedError
        elif self.tokenize == 'deberta':
            if self.language == 'en':
                self.tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
            elif self.language == 'zh':
                self.tokenizer = AutoTokenizer.from_pretrained('./cache/Erlangshen-DeBERTa-v2-97M-Chinese', use_fast=False)
            else:
                raise NotImplementedError
        elif self.tokenize == 'roberta':
            if self.language == 'en':
                self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            elif self.language == 'zh':
                self.tokenizer = BertTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment')
            else:
                raise NotImplementedError
        elif self.tokenize == 'gpt2':
            if self.language == 'zh':
                self.tokenizer = GPT2Tokenizer.from_pretrained('IDEA-CCNL/Wenzhong-GPT2-110M')
            elif self.language == 'en':
                self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            else:
                raise NotImplementedError
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            raise NotImplementedError
        logger.info(f'[Loaded Tokenizer: {self.tokenize} in language {self.language}]')

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
            context = conv_dict['context']
            batch_context.append(context)

            if 'context_goals' not in conv_dict:
                batch_context_goal.append("")
            else:
                context_goals = conv_dict['context_goals']
                context_goals = '; '.join(context_goals)
                batch_context_goal.append(context_goals)

            if 'user_profile' not in conv_dict:
                batch_user_profile.append("")
            else:
                user_profile = conv_dict['user_profile']
                batch_user_profile.append(user_profile)

            batch_label.append(conv_dict['label'])

        
        batch_context = self.tokenizer(batch_context, padding=True, truncation=True, max_length=self.context_truncate, return_tensors='pt')
        batch_context_goal = self.tokenizer(batch_context_goal, padding=True, truncation=True, max_length=self.context_truncate, return_tensors='pt')
        batch_user_profile = self.tokenizer(batch_user_profile, padding=True, truncation=True, max_length=self.entity_truncate, return_tensors='pt')
        batch_label = torch.tensor(batch_label, dtype=torch.float32)

        return (batch_context['input_ids'], batch_context['attention_mask'],
                batch_context_goal['input_ids'], batch_context_goal['attention_mask'],
                batch_user_profile['input_ids'], batch_user_profile['attention_mask'], batch_label)
