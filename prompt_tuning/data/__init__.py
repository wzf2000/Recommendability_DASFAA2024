from .durecdial import DuRecDialDataset

def get_datasets(dataset_name: str, language: str, zero_shot: bool = False):
    if dataset_name == 'DuRecDial':
        if zero_shot:
            return DuRecDialDataset(language, 'test')
        else:
            return DuRecDialDataset(language, 'train'), DuRecDialDataset(language, 'dev'), DuRecDialDataset(language, 'test')
    else:
        raise NotImplementedError