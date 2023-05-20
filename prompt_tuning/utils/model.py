from transformers import AutoTokenizer, AutoModel, PreTrainedModel, PreTrainedTokenizer, T5Tokenizer, AutoConfig, BertTokenizer
from typing import Tuple
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, One2oneVerbalizer
from openprompt import PromptForClassification
from ..plms import GLMTokenizerWrapper

def get_backbone(model_name: str, language: str, size: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer, dict, type]:
    # tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    # model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
    if model_name == 'bert':
        if language == 'zh':
            assert size == 'base'
            plm, tokenizer, model_config, WrapperClass = load_plm("bert", f"bert-{size}-chinese")
        else:
            assert size in ['base', 'large']
            plm, tokenizer, model_config, WrapperClass = load_plm("bert", f"bert-{size}-cased")
    elif model_name == 't5':
        if language == 'zh':
            raise Exception(f'Unsupported language {language} for t5!')
        else:
            assert size in ['small', 'base', 'large', '3b', '11b']
            plm, tokenizer, model_config, WrapperClass = load_plm("t5", f"t5-{size}")
    elif model_name == 'gpt2':
        if language == 'zh':
            raise Exception(f'Unsupported language {language} for gpt2!')
        else:
            assert size in ['base', 'medium', 'large', 'xl']
            if size == 'base':
                plm, tokenizer, model_config, WrapperClass = load_plm("gpt2", "gpt2")
            else:
                plm, tokenizer, model_config, WrapperClass = load_plm("gpt2", f"gpt2-{size}")
    elif model_name == 'roberta':
        if language == 'zh':
            raise Exception(f'Unsupported language {language} for roberta!')
        else:
            assert size in ['base', 'large']
            plm, tokenizer, model_config, WrapperClass = load_plm("roberta", f"roberta-{size}")
    elif model_name == 'glm':
        assert size in ['base', 'medium', 'large']
        if size == 'base':
            plm = AutoModel.from_pretrained('THUDM/chatglm-6b-int4', trust_remote_code=True).half()
        elif size == 'medium':
            plm = AutoModel.from_pretrained('THUDM/chatglm-6b-int8', trust_remote_code=True).half()
        else:
            plm = AutoModel.from_pretrained('THUDM/chatglm-6b', trust_remote_code=True).half()
        model_config = AutoConfig.from_pretrained('THUDM/chatglm-6b', trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
        WrapperClass = GLMTokenizerWrapper
    else:
        raise Exception(f'Unsupported model {model_name}!')
    return plm, tokenizer, model_config, WrapperClass



def get_model(template: ManualTemplate, verbalizer: One2oneVerbalizer, plm: PreTrainedModel) -> PromptForClassification:
    return PromptForClassification(
        template=template,
        verbalizer=verbalizer,
        plm=plm
    ).cuda()