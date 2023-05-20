from transformers import PreTrainedTokenizer, T5Tokenizer
from typing import List
from loguru import logger
from openprompt import PromptDataLoader
from openprompt.prompts import ManualTemplate
from openprompt.data_utils import InputExample
from ..plms import GLMTokenizerWrapper

def get_dataloader(tokenizer: PreTrainedTokenizer, dataset: List[InputExample], template: ManualTemplate, WrapperClass, batch_size: int) -> PromptDataLoader:
    if WrapperClass == GLMTokenizerWrapper:
        loader = PromptDataLoader(
            dataset=dataset,
            tokenizer=tokenizer,
            template=template,
            batch_size=batch_size,
            tokenizer_wrapper_class=WrapperClass,
            max_seq_length=1024,
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
        )
    else:
        loader = PromptDataLoader(
            dataset=dataset,
            tokenizer=tokenizer,
            template=template,
            batch_size=batch_size,
            tokenizer_wrapper_class=WrapperClass,
            max_seq_length=1024,
        )
    return loader