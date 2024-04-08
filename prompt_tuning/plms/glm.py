from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import PaddingStrategy
from openprompt.plms.utils import TokenizerWrapper
from typing import List, Dict, Optional
from collections import defaultdict
import numpy as np
from openprompt.utils.logging import logger
import warnings


class GLMTokenizerWrapper(TokenizerWrapper):
    def __init__(self,
                 max_seq_length: int,
                 tokenizer: PreTrainedTokenizer,
                 truncate_method: Optional[str] = 'tail',
                 **kwargs):
        super().__init__(max_seq_length=max_seq_length, tokenizer=tokenizer,truncate_method=truncate_method)
        self.exceed_num = 0

    @property
    def num_special_tokens_to_add(self):
        if not hasattr(self, '_num_specials'):
            self._num_specials = self.tokenizer.num_special_tokens_to_add()
        return self._num_specials

    def tokenize_one_example(self, wrapped_example, teacher_forcing):
        '''
        '''
        assert teacher_forcing == False
        wrapped_example, others = wrapped_example

        text = ''
        for piece in wrapped_example:
            if piece['text'] == self.template_mask_token:
                break
            text += piece['text']
        
        text = f"[Round 0]\n问：{text}\n答："
        
        encoder_inputs = self.tokenizer(text, padding=PaddingStrategy.MAX_LENGTH, truncation=True, max_length=self.max_seq_length, return_attention_mask=True, return_token_type_ids=True)
        encoder_inputs['loss_ids'] = [0] * (len(encoder_inputs['input_ids']) - 1) + [1]
        if encoder_inputs['input_ids'][0] != self.tokenizer.pad_token_id:
            self.exceed_num += 1
        
        encoder_inputs.pop("position_ids")

        return encoder_inputs


