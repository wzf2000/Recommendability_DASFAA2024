from crslab.config import Config
from crslab.data.dataloader import *
from crslab.data.dataset import *
from crslab.system import *
from crslib.data.dataset import MyDuRecDialDataset
from crslib.data.dataloader import RecommendableDataLoader
from crslib.system import RecommendableSystem


def run():
    config = Config('test.yaml', '2', False)
    # get_dataset()
    dataset = MyDuRecDialDataset(opt=config, tokenize=config['tokenize'], restore=True, save=False)
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