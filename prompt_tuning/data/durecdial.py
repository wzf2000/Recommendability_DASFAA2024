from openprompt.data_utils import InputExample
from torch.utils.data import Dataset
from tqdm import tqdm
from loguru import logger
from copy import copy
from typing import List
import re
import json
from ..utils import read_txt_data

class DuRecDialDataset(Dataset):
    def __init__(self, language: str, phase: str) -> None:
        super().__init__()
        self.language = language
        self.phase = phase
        raw_data = read_txt_data(f'datasets/DuRecDial/{self.language}_{self.phase}.txt')
        self.data = self.process_data(raw_data)
        logger.info(f'[process data from {self.phase} completed]')
        logger.info(f'[{self.phase} data size: {len(self.data)}]]')
        self.dataset = []
        pos_cnt = 0
        # processed_dataset = []
        for i, sample in enumerate(self.data):
            self.dataset.append(InputExample(
                guid=i,
                text_a=sample['context'],
                label=sample['label']
            ))
            if sample['label'] == 1:
                pos_cnt += 1
            # processed_dataset.append({
            #     'context': sample['context'],
            #     'label': sample['label']
            # })
        # with open(f'datasets/DuRecDial/{self.language}_{self.phase}_processed.json', 'w', encoding='utf8') as f:
        #     json.dump(processed_dataset, f, indent=2, ensure_ascii=False)
        # logger.info(f'[save processed data to data/{self.language}_{self.phase}_processed.json completed]')
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
        def convert(conversation, language: str):
            augmented_convs = []
            if language == 'zh':
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

                augmented_convs.append({
                    'role': 'Seeker' if utt_role == 1 else 'Recommender',
                    'text': utt,
                    'goal': conversation['goal_type_list'][utt_id],
                })
                last_role = utt_role

            return augmented_convs

        def augment_data(raw_conv_dict, language):
            augmented_conv = []
            context = ''
            last_rec = ''
            for conv in raw_conv_dict:
                if len(context) > 0 and conv['role'] == 'Recommender':
                    if ('推荐' not in conv['goal'] and 'recommendation' not in conv['goal']) or conv['goal'] != last_rec:
                        conv_dict = {
                            'context' : copy(context),
                            'label': 1 if '推荐' in conv['goal'] or 'recommendation' in conv['goal'] else 0,
                        }
                        augmented_conv.append(conv_dict)
                
                if '推荐' in conv['goal'] or 'recommendation' in conv['goal']:
                    last_rec = conv['goal']

                if language == 'zh':
                    if conv['role'] == 'Seeker':
                        context += f"用户: {conv['text']}\n"
                    else:
                        context += f"系统: {conv['text']}\n"
                else:
                    if conv['role'] == 'Seeker':
                        context += f"User: {conv['text']}\n"
                    else:
                        context += f"System: {conv['text']}\n"

            return augmented_conv

        data = [convert(i, self.language) for i in tqdm(data)]
        processed_data = []
        for conv in tqdm(data):
            processed_data.extend(augment_data(conv, self.language))
        return processed_data