from crslab.data.dataset import BaseDataset
from loguru import logger
from tqdm import tqdm
from copy import copy
import os
import json
from bidict import bidict
from transformers import BertTokenizer
import numpy as np
import re

class MyDuRecDialDataset(BaseDataset):
    def __init__(self, opt, tokenize, restore=False, save=False):
        dpath = os.path.join('dataset', 'DuRecDial2.0')
        self.language = opt['language']
        self.opt = opt
        self.dpath = dpath

        if not restore:
            # load and process
            train_data, valid_data, test_data, self.vocab = self._load_data()
            logger.info('[Finish data load]')
            self.train_data, self.valid_data, self.test_data, self.side_data = self._data_preprocess(train_data,
                                                                                                     valid_data,
                                                                                                     test_data)
            embedding = opt.get('embedding', None)
            if embedding:
                self.side_data["embedding"] = np.load(os.path.join(self.dpath, embedding))
                logger.debug(f'[Load pretrained embedding {embedding}]')
            logger.info('[Finish data preprocess]')
        else:
            self.train_data, self.valid_data, self.test_data, self.side_data = self._load_from_restore(file_name=f'{self.language}_all_data.pkl')
            self.vocab = self._load_vocab()

        def count_label(dataset):
            cnt = 0
            for data in dataset:
                if data['label'] == 1:
                    cnt += 1
            return cnt

        logger.info(f"[Train data recommendable proportion = {count_label(self.train_data) / len(self.train_data) * 100:.2f}%]")
        logger.info(f"[Valid data recommendable proportion = {count_label(self.valid_data) / len(self.valid_data) * 100:.2f}%]")
        logger.info(f"[Test data recommendable proportion = {count_label(self.test_data) / len(self.test_data) * 100:.2f}%]")

        if save:
            data = (self.train_data, self.valid_data, self.test_data, self.side_data)
            self._save_to_one(data, file_name=f'{self.language}_all_data.pkl')
    
    def _load_data(self):
        train_data, valid_data, test_data = self._load_raw_data()
        vocab = self._load_vocab()
        self._load_other_data()

        return train_data, valid_data, test_data, vocab
    
    def _load_from_txt(self, file_name: str):
        with open(file_name, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            data_list = []
            for line in lines:
                data = json.loads(line)
                data_list.append(data)
            return data_list

    def _load_raw_data(self):
        train_data = self._load_from_txt(os.path.join(self.dpath, f'{self.language}_train.txt'))
        logger.debug(f"[Load train data from {os.path.join(self.dpath, f'{self.language}_train.txt')}]")
        valid_data = self._load_from_txt(os.path.join(self.dpath, f'{self.language}_dev.txt'))
        logger.debug(f"[Load valid data from {os.path.join(self.dpath, f'{self.language}_dev.txt')}]")
        test_data = self._load_from_txt(os.path.join(self.dpath, f'{self.language}_test.txt'))
        logger.debug(f"[Load test data from {os.path.join(self.dpath, f'{self.language}_test.txt')}]")

        return train_data, valid_data, test_data

    def _load_vocab(self):
        # default to use BERT
        if self.language == 'zh':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.ind2tok = self.tokenizer.ids_to_tokens
        self.tok2ind = bidict(self.ind2tok).inverse

        logger.debug(f"[Load vocab from {self.tokenizer.name_or_path}]")
        logger.debug(f"[The size of token2index dictionary is {len(self.tok2ind)}]")
        logger.debug(f"[The size of index2token dictionary is {len(self.ind2tok)}]")

        self.special_token_idx = {
            'unk_token': self.tokenizer.unk_token_id,
            'sep_token': self.tokenizer.sep_token_id,
            'pad_token': self.tokenizer.pad_token_id,
            'cls_token': self.tokenizer.cls_token_id,
            'mask_token': self.tokenizer.mask_token_id,
        }
        self.unk_token_idx = self.special_token_idx['unk_token']
        vocab = {
            'tok2ind': self.tok2ind,
            'ind2tok': self.ind2tok,
            # 'entity2id': self.entity2id,
            # 'id2entity': self.id2entity,
            # 'word2id': self.word2id,
            'vocab_size': self.tokenizer.vocab_size,
            # 'n_entity': self.n_entity,
            # 'n_word': self.n_word,
        }
        vocab.update(self.special_token_idx)
        return vocab

    def _load_other_data(self):
        #! No other data now
        pass

    def _data_preprocess(self, train_data, valid_data, test_data):
        processed_train_data = self._raw_data_process(train_data)
        logger.debug("[Finish train data process]")
        processed_valid_data = self._raw_data_process(valid_data)
        logger.debug("[Finish valid data process]")
        processed_test_data = self._raw_data_process(test_data)
        logger.debug("[Finish test data process]")
        processed_side_data = self._side_data_process()
        logger.debug("[Finish side data process]")
        return processed_train_data, processed_valid_data, processed_test_data, processed_side_data

    def _raw_data_process(self, raw_data):
        augmented_convs = [self._convert_to_id(conversation) for conversation in tqdm(raw_data)]
        augmented_conv_dicts = []
        for conv in tqdm(augmented_convs):
            augmented_conv_dicts.extend(self._augment_and_add(conv))
        return augmented_conv_dicts

    def _convert_to_id(self, conversation):
        augmented_convs = []
        if self.language == 'zh':
            # 0: Recommender, 1: Seeker
            if conversation['goal_type_list'][0] == '寒暄':
                last_role = 1
            else:
                last_role = 0
        else:
            if conversation['goal_type_list'][0] == 'Greetings':
                last_role = 1
            else:
                last_role = 0
        for utt_id, utt in enumerate(conversation['conversation']):
            utt_role = last_role ^ 1

            res = re.findall(r'\[[0-9]*\]', utt)
            prefix = 0 if len(res) == 0 else len(res[0])
            utt = utt[prefix:]
            text_token_ids = self.tokenizer.encode(utt)[1:]

            user_profile: dict = conversation['user_profile']
            user_profile = [f"我的 {key}: {','.join(value)}" if isinstance(value, list) else f"我的 {key}: {value}" for key, value in user_profile.items()]
            user_profile = [self.tokenizer.encode(sent)[1:] for sent in user_profile]

            augmented_convs.append({
                'role': 'Seeker' if utt_role == 1 else 'Recommender',
                'text': text_token_ids,
                'user_profile': user_profile,
                'goal': conversation['goal_type_list'][utt_id],
            })
            last_role = utt_role

        return augmented_convs

    def _augment_and_add(self, raw_conv_dict):
        augmented_conv_dicts = []
        context_tokens = []
        context_goals = []
        last_rec = ''
        for conv in raw_conv_dict:
            text_tokens = conv['text']
            if len(context_tokens) > 0 and conv['role'] == 'Recommender':
                if '推荐' not in conv['goal'] or conv['goal'] != last_rec:
                    conv_dict = {
                        'role': conv['role'],
                        'context_tokens': copy(context_tokens),
                        'context_goals': [self.tokenizer.encode(context_goal)[1:] for context_goal in context_goals],
                        'response': text_tokens,
                        'user_profile': conv['user_profile'],
                        'label': 1 if '推荐' in conv['goal'] or 'recommendation' in conv['goal'] else 0,
                    }
                    augmented_conv_dicts.append(conv_dict)
            
            if '推荐' in conv['goal']:
                last_rec = conv['goal']

            context_tokens.append(text_tokens)
            context_goals.append(conv['goal'])

        return augmented_conv_dicts

    def _side_data_process(self):
        #! No side data now
        return {}
