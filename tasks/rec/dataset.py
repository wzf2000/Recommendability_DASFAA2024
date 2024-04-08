import torch
from torch.utils import data
from torch.utils.data import Dataset
from datasets.arrow_dataset import Dataset as HFDataset
from datasets.load import load_dataset
from datasets import DatasetDict, Dataset
from evaluate import combine
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    default_data_collator,
)
import numpy as np
import logging

logger = logging.getLogger(__name__)

import json

def read_data(filepath: str):
    with open(filepath, 'r', encoding='utf8') as f:
        data = json.load(f)
    return data

class RecDataset():
    def __init__(self, tokenizer: AutoTokenizer, data_args, training_args) -> None:
        super().__init__()
        self.raw_datasets = DatasetDict({
            'train': Dataset.from_list(read_data(f'{self.get_dir()}/{data_args.dataset_language}_train_processed.json')),
            'dev': Dataset.from_list(read_data(f'{self.get_dir()}/{data_args.dataset_language}_dev_processed.json')),
            'test': Dataset.from_list(read_data(f'{self.get_dir()}/{data_args.dataset_language}_test_processed.json'))
        })
        self.tokenizer = tokenizer
        self.data_args = data_args
        #labels
        self.is_regression = False
        self.label_list = [0, 1]
        self.num_labels = 2

        # Padding strategy
        if data_args.pad_to_max_length:
            self.padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            self.padding = False

        self.label2id = {l: i for i, l in enumerate(self.label_list)}
        self.id2label = {id: label for label, id in self.label2id.items()}
        
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        self.max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

        self.raw_datasets = self.raw_datasets.map(
            self.preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenization on dataset",
        )

        if training_args.do_train:
            self.train_dataset = self.raw_datasets["train"]
            if data_args.max_train_samples is not None:
                self.train_dataset = self.train_dataset.select(range(data_args.max_train_samples))

        if training_args.do_eval:
            self.eval_dataset = self.raw_datasets["dev"]
            if data_args.max_eval_samples is not None:
                self.eval_dataset = self.eval_dataset.select(range(data_args.max_eval_samples))

        if training_args.do_predict or data_args.dataset_name is not None or data_args.test_file is not None:
            self.predict_dataset = self.raw_datasets["test"]
            if data_args.max_predict_samples is not None:
                self.predict_dataset = self.predict_dataset.select(range(data_args.max_predict_samples))

        self.metric = combine(["f1", "accuracy", "precision", "recall"])

        if data_args.pad_to_max_length:
            self.data_collator = default_data_collator
        elif training_args.fp16:
            self.data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    def get_dir(self):
        raise NotImplementedError

    def preprocess_function(self, examples):
        # Tokenize the texts
        args = (
            (examples['context'],)
        )
        result = self.tokenizer(*args, padding=self.padding, max_length=self.max_seq_length, truncation=True)

        return result

    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        result = self.metric.compute(predictions=preds, references=p.label_ids)
        return result


class DuRecDialDataset(RecDataset):
    def get_dir(self):
        return '../datasets/DuRecDial'

class JDDCDataset(RecDataset):
    def get_dir(self):
        return '../datasets/JDDC'

    def __init__(self, tokenizer: AutoTokenizer, data_args, training_args) -> None:
        assert data_args.dataset_language == 'zh'
        super().__init__(tokenizer, data_args, training_args)
