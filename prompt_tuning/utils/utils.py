from openprompt.prompts import One2oneVerbalizer
from argparse import ArgumentParser
from loguru import logger
import numpy as np
import torch
import random
import json
from .template import template_num

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# Write a function to read from data/*_*.txt and output the json format of the data
# Each line of the data is in json format, so you can use json.loads() to convert it to a dictionary
# The output should be a list of dictionaries, where each dictionary is a row of the data
def read_txt_data(file_name: str):
    """
    Read data from file_name and return a list of dictionaries
    """
    with open(file_name, 'r', encoding='utf8') as f:
        data = f.readlines()
    data = [json.loads(i) for i in data]
    logger.info(f'[load data from {file_name} completed]')
    return data

def read_json_data(file_name: str):
    """
    Read data from file_name and return a list of dictionaries
    """
    with open(file_name, 'r', encoding='utf8') as f:
        data = json.load(f)
    logger.info(f'[load data from {file_name} completed]')
    return data

def post_log_softmax(verbalizer: One2oneVerbalizer, logits: torch.Tensor) -> torch.Tensor:
    # normalize
    probs = verbalizer.normalize(logits)

    # calibrate
    if  hasattr(verbalizer, "_calibrate_logits") and verbalizer._calibrate_logits is not None:
        probs = verbalizer.calibrate(label_words_probs=probs)

    # convert to logits
    logits = torch.log(probs+1e-15)
    return logits

def parse():
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument('-d', '--dataset', type=str, default='DuRecDial', choices=['DuRecDial', 'JDDC'], help='dataset name')
    parser.add_argument('-l', '--language', type=str, default='zh', choices=['zh', 'en'], help='The language of dataset (only for DuRecDial)')
    parser.add_argument('--split', type=int, default=1, choices=[1], help='The split of dataset (only for JDDC)')
    parser.add_argument('-m', '--model', type=str, default='bert', choices=['bert', 't5', 'gpt2', 'roberta', 'glm', 'glm2'], help='backbone name')
    parser.add_argument('-s', '--size', type=str, default='base', choices=['small', 'base', 'medium', 'large', 'xl', '3b', '11b'], help='backbone size')
    parser.add_argument('-z', '--zero_shot', action='store_true', help='Whether or not to finetune the model')
    parser.add_argument('-f', '--few_shot', action='store_true', help='Whether to use few-shot learning')
    parser.add_argument('--new', action='store_true', help='Whether to use new experiment setting')
    parser.add_argument('--balance', action='store_true', help='Whether to balance the training set when using few-shot learning')
    parser.add_argument('--few_shot_num', type=int, default=30, help='The number of few-shot examples per class when using few-shot learning')
    parser.add_argument('--times', type=int, default=1, help='The times every epoch is repeated when using few-shot learning')
    parser.add_argument('-e', '--epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('-w', '--weight_decay', type=float, default=1e-2, help='weight decay')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('-t', '--template', type=int, default=0, choices=range(template_num), help='template id')
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'wce', 'focal', 'dsc'], help='loss function')
    parser.add_argument('--alpha', type=float, default=1.0, help='alpha for wce and focal')
    parser.add_argument('--gamma', type=float, default=0.0, help='gamma for focal and dsc')
    parser.add_argument('--smooth', type=float, default=1e-4, help='smooth for dsc')
    parser.add_argument('--dice_square', action='store_true', help='whether to square the dice')
    args = parser.parse_args()
    return args