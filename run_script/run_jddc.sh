export CUDA_VISIBLE_DEVICES=4
export DATASET_NAME=jddc
export psl=40

# zh
export language=zh
bs=16 model=bert-base-chinese log_name=bert-base bash run_script/run_rec.sh
bs=16 model=IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment log_name=roberta-base bash run_script/run_rec.sh
bs=4 model=IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment log_name=roberta-large bash run_script/run_rec.sh
