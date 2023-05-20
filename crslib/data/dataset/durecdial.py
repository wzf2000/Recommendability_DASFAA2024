from crslab.data.dataset import BaseDataset
from crslab.data.dataset.durecdial.resources import resources
from crslab.config import DATASET_PATH
from crslab.download import build
from loguru import logger
from tqdm import tqdm
from copy import copy
import os
import json
from bidict import bidict
from transformers import AutoTokenizer, BertTokenizer, GPT2Tokenizer, DebertaTokenizer, RobertaTokenizer
import numpy as np
import re

class MyDuRecDialDataset(BaseDataset):
    def __init__(self, opt, tokenize, restore=False, save=False):
        dpath = os.path.join('dataset', 'DuRecDial2.0')
        self.language = opt['language']
        self.opt = opt
        self.tokenize = tokenize
        self.dpath = dpath

        if not restore:
            # load and process
            train_data, valid_data, test_data = self._load_data()
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
            self.train_data, self.valid_data, self.test_data, self.side_data = self._load_from_restore(file_name=f"{self.language}_{self.tokenize}_all_data.pkl")

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
            self._save_to_one(data, file_name=f"{self.language}_{self.tokenize}_all_data.pkl")
    
    def _load_data(self):
        train_data, valid_data, test_data = self._load_raw_data()
        self._load_other_data()

        return train_data, valid_data, test_data
    
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
            # text_token_ids = self._tokenize_text(utt)[1:]

            user_profile: dict = conversation['user_profile']
            user_profile = [f"我的 {key}: {','.join(value)}" if isinstance(value, list) else f"我的 {key}: {value}" for key, value in user_profile.items()]
            user_profile = '\n'.join(user_profile)
            # user_profile = [self._tokenize_text(sent)[1:] for sent in user_profile]

            augmented_convs.append({
                'role': 'Seeker' if utt_role == 1 else 'Recommender',
                'text': utt,
                'user_profile': user_profile,
                'goal': conversation['goal_type_list'][utt_id],
            })
            last_role = utt_role

        return augmented_convs

    def _augment_and_add(self, raw_conv_dict):
        augmented_conv_dicts = []
        context = ''
        context_goals = []
        last_rec = ''
        for conv in raw_conv_dict:
            text = conv['text']
            if len(context) > 0 and conv['role'] == 'Recommender':
                if ('推荐' not in conv['goal'] and 'recommendation' not in conv['goal']) or conv['goal'] != last_rec:
                    conv_dict = {
                        'role': conv['role'],
                        'context': copy(context),
                        'context_goals': copy(context_goals),
                        'response': copy(text),
                        'user_profile': conv['user_profile'],
                        'label': 1 if '推荐' in conv['goal'] or 'recommendation' in conv['goal'] else 0,
                    }
                    augmented_conv_dicts.append(conv_dict)
            
            if '推荐' in conv['goal'] or 'recommendation' in conv['goal']:
                last_rec = conv['goal']

            if self.language == 'zh':
                if conv['role'] == 'Seeker':
                    context += f"用户: {conv['text']}\n"
                else:
                    context += f"系统: {conv['text']}\n"
            else:
                if conv['role'] == 'Seeker':
                    context += f"User: {conv['text']}\n"
                else:
                    context += f"System: {conv['text']}\n"
            context_goals.append(conv['goal'])

        return augmented_conv_dicts

    def _side_data_process(self):
        #! No side data now
        return {}
