from evaluate import combine
from openprompt import PromptForClassification, PromptDataLoader
from openprompt.data_utils import InputExample
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import AdamW
import torch
from torch.utils.data import random_split
import numpy as np
from tqdm import tqdm
from loguru import logger
from typing import List
import random

from prompt_tuning.utils import *
from prompt_tuning.data import *
from prompt_tuning.loss import *

metric = combine(["accuracy", "f1", "precision", "recall"])
logger.info("Metric loading complete.")

def get_log_name(args):
    return f'{args.dataset}/{args.language if args.dataset == "DuRecDial" else args.split}_{args.model}-{args.size}_seed={args.seed}_tempate={args.template}{"_loss=" + args.loss if args.loss != "ce" else ""}{"_z" if args.zero_shot else ""}{"_f" if args.few_shot else ""}{"_new" if args.new else ""}'

def get_loss_tag(args):
    if args.loss == 'ce':
        return 'ce'
    elif args.loss == 'wce':
        return f'wce-{args.alpha}'
    elif args.loss == 'focal':
        return f'focal-{args.alpha}-{args.gamma}'
    elif args.loss == 'dsc':
        return f'dsc-{args.gamma}-{args.smooth}-{"square" if args.dice_square else "linear"}'

def get_shot_tag(args):
    if args.zero_shot:
        return 'zero-shot'
    elif args.few_shot:
        return f'few-shot-{args.times}-{"balance" if args.balance else "unbalance"}'
    else:
        return 'fine-tune'

# Cacluate the accuracy, precision and recall of the model
def evaluate(model: PromptForClassification, dataloader: PromptDataLoader, opt, if_predict: bool = False):
    logger.info("Evaluating model...")
    model.eval()
    refs = []
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: v.cuda() for k, v in batch.items()}
            logits = model(batch)
            pred = torch.argmax(logits, dim=-1)
            for i, (predict, label) in enumerate(zip(pred, batch['label'].cpu())):
                refs.append(label)
                predictions.append(predict)

    result = metric.compute(predictions=predictions, references=refs)
    logger.info("Evaluate result:")
    logger.info(f"Evaluate F1: {result['f1']}")
    logger.info(f"Evaluate Precision: {result['precision']}")
    logger.info(f"Evaluate Recall: {result['recall']}")
    logger.info(f"Evaluate Accuracy: {result['accuracy']}")
    logger.info(f"Format: Accuracy\tPrecision\tRecall\tF1")
    logger.info(f"{result['accuracy']:.4f}\t{result['precision']:.4f}\t{result['recall']:.4f}\t{result['f1']:.4f}")
    if if_predict:
        with open(f'log/{opt.dataset}/result_merge.csv', 'a') as f:
            f.write(f"{opt.language}\t{opt.model}-{opt.size}-{opt.seed}\t{opt.template}\t{get_loss_tag(opt)}\t{get_shot_tag(opt)}\t{result['accuracy']:.4f}\t{result['precision']:.4f}\t{result['recall']:.4f}\t{result['f1']:.4f}\n")
    return result['f1']

def train(model: PromptForClassification, dataloader: PromptDataLoader, val_dataloader: PromptDataLoader, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LambdaLR, opt):
    best_metric = 0
    best_epoch = -1
    if opt.loss == "ce":
        loss_func = torch.nn.CrossEntropyLoss()
    elif opt.loss == "wce":
        loss_func = torch.nn.CrossEntropyLoss(weight=torch.tensor([opt.alpha, 1.0]).cuda())
    elif opt.loss == "focal":
        loss_func = MultiFocalLoss(2, alpha=torch.tensor([opt.alpha, 1.0]).cuda(), gamma=opt.gamma)
    elif opt.loss == "dsc":
        loss_func = MultiDSCLoss(gamma=opt.gamma, smooth=opt.smooth, dice_square=opt.dice_square)
    torch.save(model.state_dict(), f'checkpoint/{get_log_name(opt)}_best_model.pt')
    for epoch in range(opt.epochs):
        model.train()
        times = opt.times if opt.few_shot else 1
        steps = len(dataloader) * times
        acc_merge = []
        with tqdm(total=steps, leave=False) as pbar:
            for _ in range(times):
                for batch in dataloader:
                    batch = {k: v.cuda() for k, v in batch.items()}
                    logits = model(batch)
                    # logits = post_log_softmax(model.verbalizer, logits)
                    preds = torch.argmax(logits, dim=-1)
                    acc = torch.sum(preds == batch['label']).item() / len(preds)
                    acc_merge.append(acc)
                    loss = loss_func(logits, batch['label'])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    pbar.update(1)
                    pbar.set_description(f"Epoch {epoch} loss: {loss.item():.4f} acc: {np.mean(acc_merge):.4f}")
                    pbar.set_postfix(lr=optimizer.param_groups[0]['lr'])
        logger.info(f"Epoch {epoch} finished, average acc: {np.mean(acc_merge):.4f}")
        metric = evaluate(model, val_dataloader, opt)
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            logger.info(f"New best metric: {best_metric}")
            torch.save(model.state_dict(), f'checkpoint/{get_log_name(opt)}_best_model.pt')
        # early stop
        if epoch - best_epoch >= 3:
            break

def main():
    args = parse()
    set_seed(args.seed)
    logger.add(f'log/{get_log_name(args)}.log', rotation="500 MB", level="INFO")
    datasets = get_datasets(args.dataset, args.language, args.split, args.zero_shot)
    logger.info(f'[task loading completed]')
    plm, tokenizer, model_config, WrapperClass = get_backbone(args.model, args.language, args.size)
    logger.info(f'[backbone loading completed]')
    template = get_template(tokenizer, args.language, args.template, args.model)
    logger.info(f'[template loading completed]')
    verbalizer = get_verbalizer(tokenizer, classes, args.model, args.language)
    logger.info(f'[verbalizer loading completed]')
    model = get_model(template, verbalizer, plm)
    logger.info(f'[model loading completed]')
    if args.zero_shot:
        dataloader = get_dataloader(tokenizer, datasets, template, WrapperClass, args.batch_size, train=False)
        logger.info(f'[data loading completed]')
        evaluate(model, dataloader, args, if_predict=True)
    else:
        if args.few_shot:
            if not args.balance:
                generator = torch.Generator().manual_seed(2023)
                datasets[0].dataset = random_split(datasets[0].dataset, [args.few_shot_num, len(datasets[0]) - args.few_shot_num], generator=generator)[0]
            else:
                train_dataset: List[InputExample] = datasets[0].dataset
                pos_dataset = [example for example in train_dataset if example.label == 1]
                neg_dataset = [example for example in train_dataset if example.label == 0]
                random.shuffle(pos_dataset)
                random.shuffle(neg_dataset)
                pos_sample_num = args.few_shot_num // 2
                neg_sample_num = args.few_shot_num - pos_sample_num
                datasets[0].dataset = pos_dataset[:pos_sample_num] + neg_dataset[:neg_sample_num]
        else:
            assert args.times == 1

        dataloaders = [get_dataloader(tokenizer, dataset, template, WrapperClass, args.batch_size, train=(i == 0)) for (i, dataset) in enumerate(datasets)]
        logger.info(f'[data loading completed]')
        if args.new:
            if args.size == 'large':
                param_optimizer = list(model.named_parameters())
                no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
                    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]
                optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
            else:
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            tot_step  = len(dataloaders[0]) * args.epochs * args.times
            if args.size == 'large':
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(tot_step * 0.1), num_training_steps=tot_step)
            else:
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=tot_step)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1 / (1 + 0.05 * x))
        train(model, dataloaders[0], dataloaders[1], optimizer, scheduler, args)
        logger.info(f'[training completed]')
        # load the best model
        model.load_state_dict(torch.load(f'checkpoint/{get_log_name(args)}_best_model.pt'))
        evaluate(model, dataloaders[2], args, True)

if __name__ == "__main__":
    main()