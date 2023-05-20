from .durecdial import DuRecDialDataset
from .jddc import JDDCDataset

def get_datasets(dataset_name: str, language: str, split: int, zero_shot: bool = False):
    if dataset_name == 'DuRecDial':
        if zero_shot:
            return DuRecDialDataset(language, 'test')
        else:
            return DuRecDialDataset(language, 'train'), DuRecDialDataset(language, 'dev'), DuRecDialDataset(language, 'test')
    elif dataset_name == 'JDDC':
        assert language == 'zh'
        if zero_shot:
            return JDDCDataset(split, 'test')
        else:
            return JDDCDataset(split, 'train'), JDDCDataset(split, 'dev'), JDDCDataset(split, 'test')
    else:
        raise NotImplementedError