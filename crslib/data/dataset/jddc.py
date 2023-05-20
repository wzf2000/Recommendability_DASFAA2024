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
            self.train_data, self.valid_data, self.test_data, self.side_data = self._load_from_restore(file_name=f"{self.language}_{self.tokenize}_all_data.pkl")
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
            self._save_to_one(data, file_name=f"{self.language}_{self.tokenize}_all_data.pkl")
    
    def _load_data(self):
        train_data, valid_data, test_data = self._load_raw_data()
        vocab = self._load_vocab()
        self._load_other_data()

        return train_data, valid_data, test_data, vocab
    
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

    def _load_vocab(self):
        # default to use BERT
        if self.tokenize == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        elif self.tokenize == 'deberta':
            self.tokenizer = AutoTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-DeBERTa-v2-186M-Chinese-SentencePiece', use_fast=False, cache_dir='./cache/')
        elif self.tokenize == 'roberta':
            # self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
            self.tokenizer = BertTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment')
        elif self.tokenize == 'gpt2':
            resource = resources['gpt2']
            self.special_token_idx = resource['special_token_idx']
            self.unk_token_idx = self.special_token_idx['unk']
            dpath = os.path.join(DATASET_PATH, 'durecdial', 'gpt2')
            dfile = resource['file']
            build(dpath, dfile, version=resource['version'])
            self.tok2ind = json.load(open(os.path.join(dpath, 'token2id.json'), 'r', encoding='utf-8'))
            self.ind2tok = {idx: word for word, idx in self.tok2ind.items()}

            logger.debug(f"[Load vocab from {os.path.join(dpath, 'token2id.json')}]")
            logger.debug(f"[The size of token2index dictionary is {len(self.tok2ind)}]")
            logger.debug(f"[The size of index2token dictionary is {len(self.ind2tok)}]")
        else:
            raise NotImplementedError
        if self.tokenize == 'gpt2':
            pass
        else:
            if self.tokenize == 'bert':
                self.ind2tok = self.tokenizer.ids_to_tokens
                self.tok2ind = bidict(self.ind2tok).inverse
            else:
                self.tok2ind = self.tokenizer.get_vocab()
                self.ind2tok = bidict(self.tok2ind).inverse

            logger.debug(f"[Load vocab from {self.tokenizer.name_or_path}]")
            logger.debug(f"[The size of token2index dictionary is {len(self.tok2ind)}]")
            logger.debug(f"[The size of index2token dictionary is {len(self.ind2tok)}]")

            if self.tokenize == 'bert' or self.tokenize == 'deberta' or self.tokenize == 'roberta':
                self.special_token_idx = {
                    'pad': self.tokenizer.pad_token_id,
                    'start': self.tokenizer.cls_token_id,
                    'end': self.tokenizer.sep_token_id,
                    'unk': self.tokenizer.unk_token_id,
                    'cls': self.tokenizer.cls_token_id,
                    'sep': self.tokenizer.sep_token_id,
                    'pad_entity': self.tokenizer.pad_token_id,
                    'pad_word': self.tokenizer.pad_token_id,
                    'pad_topic': self.tokenizer.pad_token_id,
                }
            elif self.tokenize == 'gpt2':
                self.special_token_idx = {
                    'pad': self.tokenizer.eos_token_id,
                    'start': self.tokenizer.bos_token_id,
                    'end': self.tokenizer.eos_token_id,
                    'unk': self.tokenizer.unk_token_id,
                    'cls': self.tokenizer.bos_token_id,
                    'sep': self.tokenizer.eos_token_id,
                    'pad_entity': self.tokenizer.eos_token_id,
                    'pad_word': self.tokenizer.eos_token_id,
                    'pad_topic': self.tokenizer.eos_token_id,
                }
            else:
                raise NotImplementedError
            self.unk_token_idx = self.special_token_idx['unk']
        vocab = {
            'tok2ind': self.tok2ind,
            'ind2tok': self.ind2tok,
            'vocab_size': len(self.tok2ind),
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
    
    def _tokenize_text(self, text):
        if hasattr(self, 'tokenizer'):
            return self.tokenizer.encode(text)
        text = text.replace(' ', '')
        return [self.special_token_idx['start']] + [self.tok2ind.get(word, self.unk_token_idx) for word in text] + [self.special_token_idx['end']]


    def _convert_to_id(self, conversation):
        augmented_convs = []
        for utt_id, utt in enumerate(conversation['dialogs']):
            utt_role = conversation['roles'][utt_id]

            text_token_ids = [self._tokenize_text(utt)][1:]

            augmented_convs.append({
                'role': utt_role,
                'text': text_token_ids,
                'label': conversation['binary_labels'][utt_id],
            })

        return augmented_convs

    def _augment_and_add(self, raw_conv_dict):
        augmented_conv_dicts = []
        context_tokens = []
        last_role = None
        for conv in raw_conv_dict:
            text_tokens = conv['text']
            if len(context_tokens) > 0 and conv['role'] == '客服' and last_role == '顾客':
                conv_dict = {
                    'role': conv['role'],
                    'context_tokens': copy(context_tokens),
                    'response': text_tokens,
                    'label': conv['label'],
                }
                augmented_conv_dicts.append(conv_dict)

            context_tokens.append(text_tokens)
            last_role = conv['role']

        return augmented_conv_dicts

    def _side_data_process(self):
        #! No side data now
        return {}
