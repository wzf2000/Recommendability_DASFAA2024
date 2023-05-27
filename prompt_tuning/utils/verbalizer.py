from transformers import PreTrainedTokenizer
from typing import List
from openprompt.prompts import One2oneVerbalizer
from ..prompts import GLMVerbalizer

def get_verbalizer(tokenizer: PreTrainedTokenizer, classes: List[str], model_name: str, language: str) -> One2oneVerbalizer:
    if model_name == 'glm':
        return GLMVerbalizer(
            classes=classes,
            label_words={
                'no': '0',
                'yes': '1',
            },
            tokenizer=tokenizer,
            post_log_softmax=False,
        )
    elif model_name == 't5' and language == 'en':
        # T5 has a special token for 0
        # use 'zero' and 'one' as the label words
        return One2oneVerbalizer(
            classes=classes,
            label_words={
                'no': 'zero',
                'yes': 'one',
            },
            tokenizer=tokenizer,
            post_log_softmax=False,
        )
    elif model_name == 'gpt2' and language == 'zh':
        # GPT2 has a special token for 0 and 1
        # use 'a' and 'b' as the label words
        return One2oneVerbalizer(
            classes=classes,
            label_words={
                'no': 'a',
                'yes': 'b',
            },
            tokenizer=tokenizer,
            post_log_softmax=False,
        )
    else:
        return One2oneVerbalizer(
            classes=classes,
            label_words={
                'no': '0',
                'yes': '1',
            },
            tokenizer=tokenizer,
            post_log_softmax=False,
        )