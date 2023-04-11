import os
import torch
from crslab.system.base import BaseSystem
from crslab.evaluator.metrics import AverageMetric
from crslab.config import SAVE_PATH
from loguru import logger

from ..model import get_model
from ..evaluator import PolicyEvaluator

class RecommendableSystem(BaseSystem):
    def __init__(self, opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data, restore_system=False,
                 interact=False, debug=False, tensorboard=False):
        self.opt = opt
        if opt["gpu"] == [-1]:
            self.device = torch.device('cpu')
        elif len(opt["gpu"]) == 1:
            self.device = torch.device('cuda', opt["gpu"][0])
        else:
            self.device = torch.device('cuda')
        # data
        if debug:
            self.train_dataloader = valid_dataloader
        else:
            self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.vocab = vocab
        self.side_data = side_data
        # model
        if 'model' in opt:
            self.model = get_model(opt, opt['model'], self.device, vocab, side_data).to(self.device)
        else:
            if 'rec_model' in opt:
                self.rec_model = get_model(opt, opt['rec_model'], self.device, vocab['rec'], side_data['rec']).to(
                    self.device)
            if 'conv_model' in opt:
                self.conv_model = get_model(opt, opt['conv_model'], self.device, vocab['conv'], side_data['conv']).to(
                    self.device)
            if 'policy_model' in opt:
                self.policy_model = get_model(opt, opt['policy_model'], self.device, vocab['policy'],
                                              side_data['policy']).to(self.device)
        model_file_name = opt.get('model_file', f'{opt["model_name"]}.pth')
        self.model_file = os.path.join(SAVE_PATH, model_file_name)
        if restore_system:
            self.restore_model()

        if not interact:
            self.evaluator = PolicyEvaluator(tensorboard)
        
        self.optim_opt = self.opt['policy']
        self.epoch = self.optim_opt['epoch']
        self.batch_size = self.optim_opt['batch_size']
        self.language = self.opt['language']
        self.batchs = {
            'train': 0,
            'val': 0,
            'test': 0,
        }
    
    def recommendable_evaluate(self, predict, label, batch=None):
        predict = predict.cpu()
        flag = False
        for i, (predict_value, label_goal) in enumerate(zip(predict, label)):
            self.evaluator.acc_evaluate(predict_value, label_goal, batch)
            # Write a function to translate the index sequence to the token sequence
            def index2token(index: torch.Tensor):
                return self.vocab['ind2tok'][index.detach().cpu().numpy().tolist()]
            # if flag:
            #     continue
            # if (predict_value > 0.5) ^ (label_goal == 1) == 1:
            #     print(*[index2token(utt) for utt in batch[0][i]], sep='')
            #     print(f'predict: {predict_value}, label: {label_goal}')
            #     print('-----------------')
            #     while True:
            #         test = input('>>> ')
            #         if test == 'x':
            #             flag = True
            #             break
            #         elif test != 'q':
            #             try:
            #                 exec(f'print({test})')
            #             except Exception as e:
            #                 print(str(e))
            #         else:
            #             break

    def step(self, batch, stage, mode):
        #! new task
        assert stage == 'recommendable'
        batch = [ele.to(self.device) for ele in batch]
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()
        loss, predict = self.model(batch, mode)
        if mode == "train" and loss is not None:
            loss = loss.sum()
            self.backward(loss)
        else:
            self.recommendable_evaluate(predict, batch[-1], batch)
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
            self.evaluator.optim_metrics.add("loss", AverageMetric(loss))
            if mode != 'test':
                self.evaluator.writer.add_scalar(f'{mode} batch loss', loss, self.batchs[mode])
        self.batchs[mode] += 1

    def train_recommendable(self):
        params = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        params = [{
            'params': [
                p for n, p in params
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
                self.optim_opt['weight_decay']
        }, {
            'params': [
                p for n, p in params
                if any(nd in n for nd in no_decay)
            ],
        }]
        self.init_optim(self.optim_opt, params)

        for epoch in range(self.epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Recommendable epoch {str(epoch)}]')
            # change the shuffle to True
            for batch in self.train_dataloader.get_recommendable_data(self.batch_size, shuffle=True):
                self.step(batch, stage='recommendable', mode='train')
            self.evaluator.report(epoch=epoch, mode='train')
            # val
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader.get_recommendable_data(self.batch_size, shuffle=False):
                    self.step(batch, stage='recommendable', mode='val')
                self.evaluator.report(epoch=epoch, mode='val')
                # early stop
                metric = self.evaluator.acc_metrics['Accuracy']
                if self.early_stop(metric):
                    break
        # test
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader.get_recommendable_data(self.batch_size, shuffle=False):
                self.step(batch, stage='recommendable', mode='test')
            self.evaluator.report(mode='test')

    def fit(self):
        self.train_recommendable()

    def interact(self):
        pass
        # TODO: interact