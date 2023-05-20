from evaluate import combine
from openprompt import PromptForClassification, PromptDataLoader
import torch
from tqdm import tqdm
from loguru import logger

from prompt_tuning.utils import *
from prompt_tuning.data import *

# Cacluate the accuracy, precision and recall of the model
def evaluate(model: PromptForClassification, dataloader: PromptDataLoader):
    logger.info("Evaluating model...")
    model.eval()
    metric = combine(["accuracy", "f1", "precision", "recall", "roc_auc"])
    logger.info("Metric loading complete.")
    refs = []
    predition_scores = []
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: v.cuda() for k, v in batch.items()}
            logits = model(batch)
            logits = model.verbalizer.normalize(logits).cpu()
            pred = torch.argmax(logits, dim=-1)
            for i, (predict, label) in enumerate(zip(pred, batch['label'].cpu())):
                refs.append(label)
                predition_scores.append(logits[i][1])
                predictions.append(predict)

    result = metric.compute(prediction_scores=predition_scores, predictions=predictions, references=refs)
    logger.info("Evaluate result:")
    logger.info(f"Evaluate ROC AUC: {result['roc_auc']}")
    logger.info(f"Evaluate F1: {result['f1']}")
    logger.info(f"Evaluate Precision: {result['precision']}")
    logger.info(f"Evaluate Recall: {result['recall']}")
    logger.info(f"Evaluate Accuracy: {result['accuracy']}")
    return result['accuracy']

def train(model: PromptForClassification, dataloader: PromptDataLoader, val_dataloader: PromptDataLoader, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LambdaLR, opt):
    best_acc = 0
    best_epoch = -1
    loss_func = torch.nn.CrossEntropyLoss()
    for epoch in range(5):
        model.train()
        for batch in tqdm(dataloader):
            batch = {k: v.cuda() for k, v in batch.items()}
            logits = model(batch)
            logits = post_log_softmax(model.verbalizer, logits)
            loss = loss_func(logits, batch['label'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        logger.info(f"Epoch {epoch} finished, evaluating...")
        acc = evaluate(model, val_dataloader)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            torch.save(model.state_dict(), f'checkpoint/{opt.language}_{opt.model}-{opt.size}_best_model.pt')
        # early stop
        if epoch - best_epoch >= 3:
            break

def main():
    args = parse()
    datasets = get_datasets(args.dataset, args.language, args.zero_shot)
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
        dataloader = get_dataloader(tokenizer, datasets, template, WrapperClass, args.batch_size)
        logger.info(f'[data loading completed]')
        evaluate(model, dataloader)
    else:
        dataloaders = [get_dataloader(tokenizer, dataset, template, WrapperClass, args.batch_size) for dataset in datasets]
        logger.info(f'[data loading completed]')
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1 / (1 + 0.05 * x))
        train(model, dataloaders[0], dataloaders[1], optimizer, scheduler, args)
        logger.info(f'[training completed]')
        # load the best model
        model.load_state_dict(torch.load(f'checkpoint/{args.language}_{args.model}-{args.size}_best_model.pt'))
        evaluate(model, dataloaders[2])

if __name__ == "__main__":
    main()