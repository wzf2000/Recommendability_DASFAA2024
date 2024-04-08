from openprompt.data_utils import InputExample
from torch.utils.data import Dataset
from tqdm import tqdm
from loguru import logger
from copy import copy
from typing import List
import re
import json
from ..utils import read_json_data

class JDDCDataset(Dataset):
    def __init__(self, split: int, phase: str) -> None:
        super().__init__()
        self.split = split
        self.phase = phase
        raw_data = read_json_data(f'datasets/JDDC/{self.split}_{self.phase}.json')
        self.data = self.process_data(raw_data)
        logger.info(f'[process data from {self.phase} completed]')
        logger.info(f'[{self.phase} data size: {len(self.data)}]]')
        self.dataset = []
        pos_cnt = 0
        processed_dataset = []
        for i, sample in enumerate(self.data):
            self.dataset.append(InputExample(
                guid=i,
                text_a=sample['context'],
                label=sample['label']
            ))
            if sample['label'] == 1:
                pos_cnt += 1
            processed_dataset.append({
                'context': sample['context'],
                'label': sample['label']
            })
        with open(f'datasets/JDDC/zh_{self.phase}_processed.json', 'w', encoding='utf8') as f:
            json.dump(processed_dataset, f, indent=2, ensure_ascii=False)
        logger.info(f'[save processed data to data/zh_{self.phase}_processed.json completed]')
        logger.info(f'[pos_cnt: {pos_cnt}]')

    def __getitem__(self, index: int) -> InputExample:
        return self.dataset[index]

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self) -> int:
        return len(self.dataset)

    def process_data(self, data: List[dict]):
        """
        Process the data and return a list of dictionaries
        """
        def convert(conversation):
            augmented_convs = []
            for utt_id, utt in enumerate(conversation['dialogs']):
                utt_role = conversation['roles'][utt_id]
                augmented_convs.append({
                    'role': utt_role,
                    'text': utt,
                    'label': conversation['binary_labels'][utt_id],
                })
            return augmented_convs

        def augment_data(raw_conv_dict):
            augmented_conv_dicts = []
            context = ''
            last_role = None
            last_label = None
            for conv in raw_conv_dict:
                if len(context) > 0 and conv['role'] == '客服' and last_role == '顾客':
                    if last_label == conv['label'] and conv['label'] != 1:
                        augmented_conv_dicts.pop()
                    conv_dict = {
                        'context': copy(context),
                        'label': conv['label'],
                    }
                    augmented_conv_dicts.append(conv_dict)
                    last_label = conv['label']

                context += f"{conv['role']}: {conv['text']}\n"
                last_role = conv['role']
            return augmented_conv_dicts

        data = [convert(i) for i in tqdm(data)]
        processed_data = []
        for conv in tqdm(data):
            processed_data.extend(augment_data(conv))
        return processed_data