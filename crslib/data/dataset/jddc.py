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

class JDDCDataset(BaseDataset):
    def __init__(self, opt, tokenize, restore=False, save=False, split=1):
        dpath = os.path.join('dataset', 'JDDC2.1')
        self.language = opt['language']
        assert self.language == 'zh'
        self.split = split
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
    
    def _load_from_json(self, file_name: str):
        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def _load_raw_data(self):
        train_data = self._load_from_json(os.path.join(self.dpath, f'{self.split}_train.json'))
        logger.debug(f"[Load train data from {os.path.join(self.dpath, f'{self.split}_train.json')}]")
        valid_data = self._load_from_json(os.path.join(self.dpath, f'{self.split}_dev.json'))
        logger.debug(f"[Load valid data from {os.path.join(self.dpath, f'{self.split}_dev.json')}]")
        test_data = self._load_from_json(os.path.join(self.dpath, f'{self.split}_test.json'))
        logger.debug(f"[Load test data from {os.path.join(self.dpath, f'{self.split}_test.json')}]")

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
        for utt_id, utt in enumerate(conversation['dialogs']):
            utt_role = conversation['roles'][utt_id]

            # text_token_ids = self._tokenize_text(utt)[1:]

            augmented_convs.append({
                'role': utt_role,
                'text': utt,
                'label': conversation['binary_labels'][utt_id],
            })

        return augmented_convs

    def _augment_and_add(self, raw_conv_dict):
        augmented_conv_dicts = []
        context = ''
        last_role = None
        last_label = None
        for conv in raw_conv_dict:
            text = conv['text']
            if len(context) > 0 and conv['role'] == '客服' and last_role == '顾客':
                if last_label == conv['label'] and conv['label'] != 1:
                    augmented_conv_dicts.pop()
                conv_dict = {
                    'role': conv['role'],
                    'context': copy(context),
                    'response': copy(text),
                    'label': conv['label'],
                }
                augmented_conv_dicts.append(conv_dict)
                last_label = conv['label']

            context += f"{conv['role']}: {conv['text']}\n"
            last_role = conv['role']

        return augmented_conv_dicts

    def _side_data_process(self):
        #! No side data now
        return {}
