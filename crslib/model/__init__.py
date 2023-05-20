from loguru import logger
import torch

from .bert import *
from .gpt2 import *
from .deberta import *

Model_register_table = {
    'ConvBERT': ConvBERTModel,
    'ProfileBERT': ProfileBERTModel,
    'ConvGPT2': ConvGPT2Model,
    'ConvDeBERTa': ConvDeBERTaModel,
}


def get_model(config, model_name, device, vocab, side_data=None):
    if model_name in Model_register_table:
        model = Model_register_table[model_name](config, device, vocab, side_data)
        logger.info(f'[Build model {model_name}]')
        if config.opt["gpu"] == [-1]:
            return model
        else:
            if len(config.opt["gpu"]) > 1 and model_name == 'PMI':
                logger.info(f'[PMI model does not support multi GPUs yet, using single GPU now]')
                return model.to(device)
            return torch.nn.DataParallel(model, device_ids=config["gpu"])

    else:
        raise NotImplementedError('Model [{}] has not been implemented'.format(model_name))