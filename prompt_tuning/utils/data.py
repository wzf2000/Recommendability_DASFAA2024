from transformers import PreTrainedTokenizer, T5Tokenizer
from typing import List
from loguru import logger
from openprompt import PromptDataLoader
from openprompt.prompts import ManualTemplate
from openprompt.data_utils import InputExample
from ..plms import GLMTokenizerWrapper

def count_pos(dataset: List[InputExample]) -> int:
    pos = 0
    for example in dataset:
        if example.label == 1:
            pos += 1
    return pos

def get_dataloader(tokenizer: PreTrainedTokenizer, dataset: List[InputExample], template: ManualTemplate, WrapperClass, batch_size: int, train: bool) -> PromptDataLoader:
    logger.info(f"[Dataset size: {len(dataset)}]")
    pos_num = count_pos(dataset)
    logger.info(f"[Dataset positive num: {pos_num}]")
    logger.info(f"[Dataset positive ratio: {pos_num / len(dataset)}]")
    if WrapperClass == GLMTokenizerWrapper:
        loader = PromptDataLoader(
            dataset=dataset,
            tokenizer=tokenizer,
            template=template,
            batch_size=batch_size,
            tokenizer_wrapper_class=WrapperClass,
            max_seq_length=1024,
            shuffle=train,
        )
        logger.info(f'[DataLoader created: exceeded num = {loader.tokenizer_wrapper.exceed_num}]')
    elif isinstance(tokenizer, T5Tokenizer):
        loader = PromptDataLoader(
            dataset=dataset,
            tokenizer=tokenizer,
            template=template,
            batch_size=batch_size,
            tokenizer_wrapper_class=WrapperClass,
            max_seq_length=1024,
            decoder_max_length=512,
            shuffle=train,
        )
    else:
        loader = PromptDataLoader(
            dataset=dataset,
            tokenizer=tokenizer,
            template=template,
            batch_size=batch_size,
            tokenizer_wrapper_class=WrapperClass,
            max_seq_length=512,
            shuffle=train,
        )
    return loader