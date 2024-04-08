export CUDA_VISIBLE_DEVICES=3
export DATASET_NAME=durecdial
export psl=40

# zh
export language=zh
bs=16 model=bert-base-chinese log_name=bert-base bash run_script/run_rec.sh
bs=16 model=IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment log_name=roberta-base bash run_script/run_rec.sh
bs=4 model=IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment log_name=roberta-large bash run_script/run_rec.sh

# en
export language=en
bs=16 model=bert-base-cased log_name=bert-base bash run_script/run_rec.sh
bs=4 model=bert-large-cased log_name=bert-large bash run_script/run_rec.sh
bs=16 model=roberta-base log_name=roberta-base bash run_script/run_rec.sh
bs=4 model=roberta-large log_name=roberta-large bash run_script/run_rec.sh
bs=16 model=microsoft/deberta-base log_name=deberta-base bash run_script/run_rec.sh
