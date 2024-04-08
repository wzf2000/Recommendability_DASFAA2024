from argparse import ArgumentParser
from crslab.config import Config
from crslab.data.dataloader import *
from crslab.data.dataset import *
from crslab.system import *
from crslib.data.dataset import MyDuRecDialDataset, JDDCDataset
from crslib.data.dataloader import RecommendableDataLoader
from crslib.system import RecommendableSystem
import torch
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def run():
    parser = ArgumentParser()
    # Add gpu option
    parser.add_argument('--gpu', type=str, default='0', help='gpu id')
    parser.add_argument('--restore', action='store_true', help='restore dataset')
    parser.add_argument('--save', action='store_true', help='save dataset')
    parser.add_argument('--config', type=str, default='test.yaml', help='config file')
    parser.add_argument('--split', type=int, default=1, help='Dataset split number')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    # parser.add_argument('--interact', action='store_true', help='interact with model')
    args = parser.parse_args()

    set_seed(args.seed)

    config = Config(args.config, args.gpu, False)
    # get_dataset()
    if config['dataset'] == 'DuRecDial':
        dataset = MyDuRecDialDataset(opt=config, tokenize=config['tokenize'], restore=args.restore, save=args.save)
    elif config['dataset'] == 'JDDC':
        dataset = JDDCDataset(opt=config, tokenize=config['tokenize'], restore=args.restore, save=args.save, split=args.split)
    else:
        raise NotImplementedError
    
    side_data = dataset.side_data

    train_dataloader = RecommendableDataLoader(opt=config, dataset=dataset.train_data)
    valid_dataloader = RecommendableDataLoader(opt=config, dataset=dataset.valid_data)
    test_dataloader = RecommendableDataLoader(opt=config, dataset=dataset.test_data)

    CRS = RecommendableSystem(opt=config,
                              train_dataloader=train_dataloader,
                              valid_dataloader=valid_dataloader,
                              test_dataloader=test_dataloader,
                              side_data=side_data,
                              restore_system=False,
                              interact=False,
                              debug=False, 
                              tensorboard=True,
                              seed=args.seed)
    CRS.fit()

if __name__ == '__main__':
    run()