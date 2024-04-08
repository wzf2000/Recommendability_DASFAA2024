export TASK_NAME=rec

lr=5e-3
dropout=0.1
epoch=100
# compute the msl with 512 minus psl, suppose psl is not known
msl=$[512-$psl]

python3 run.py \
  --model_name_or_path $model \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --dataset_language $language \
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
  --output_dir checkpoints/$DATASET_NAME-$language-$log_name-$seed/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed $seed \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --prefix
