from argparse import ArgumentParser
from crslab.config import Config
from crslab.data.dataloader import *
from crslab.data.dataset import *
from crslab.system import *
from crslib.data.dataset import MyDuRecDialDataset, JDDCDataset
from crslib.data.dataloader import RecommendableDataLoader
from crslib.system import RecommendableSystem

def run():
    parser = ArgumentParser()
    # Add gpu option
    parser.add_argument('--gpu', type=str, default='0', help='gpu id')
    parser.add_argument('--restore', action='store_true', help='restore dataset')
    parser.add_argument('--save', action='store_true', help='save dataset')
    parser.add_argument('--config', type=str, default='test.yaml', help='config file')
    parser.add_argument('--dataset', type=str, default='durecdial', choices=['durecdial', 'jddc'], help='dataset name')
    parser.add_argument('--split', type=int, default=1, help='Dataset split number')
    # parser.add_argument('--interact', action='store_true', help='interact with model')
    args = parser.parse_args()

    config = Config(args.config, args.gpu, False)
    # get_dataset()
    if args.dataset == 'durecdial':
        dataset = MyDuRecDialDataset(opt=config, tokenize=config['tokenize'], restore=args.restore, save=args.save)
    elif args.dataset == 'jddc':
        dataset = JDDCDataset(opt=config, tokenize=config['tokenize'], restore=args.restore, save=args.save, split=args.split)
    
    side_data = dataset.side_data
    vocab = dataset.vocab

    train_dataloader = RecommendableDataLoader(opt=config, dataset=dataset.train_data, vocab=vocab)
    valid_dataloader = RecommendableDataLoader(opt=config, dataset=dataset.valid_data, vocab=vocab)
    test_dataloader = RecommendableDataLoader(opt=config, dataset=dataset.test_data, vocab=vocab)

    CRS = RecommendableSystem(opt=config,
                              train_dataloader=train_dataloader,
                              valid_dataloader=valid_dataloader,
                              test_dataloader=test_dataloader,
                              vocab=vocab,
                              side_data=side_data,
                              restore_system=False,
                              interact=False,
                              debug=False, 
                              tensorboard=True)
    CRS.fit()

if __name__ == '__main__':
    run()