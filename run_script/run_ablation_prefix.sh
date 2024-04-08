export TASK_NAME=rec
export DATASET_NAME=durecdial

bs=16
lr=5e-3
dropout=0.1
epoch=100
# compute the msl with 512 minus psl, suppose psl is not known
msl=$[512-$psl]
# output the config
echo "TASK_NAME: $TASK_NAME"
echo "DATASET_NAME: $DATASET_NAME"
echo "bs: $bs"
echo "lr: $lr"
echo "dropout: $dropout"
echo "epoch: $epoch"
echo "msl: $msl"
echo "psl: $psl"

python3 run.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --do_predict \
  --load_best_model_at_end \
  --metric_for_best_model f1 \
  --max_seq_length $msl \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/$DATASET_NAME-bert-prefix=$psl/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 11 \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --prefix
