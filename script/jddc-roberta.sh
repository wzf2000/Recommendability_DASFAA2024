export CUDA_VISIBLE_DEVICES=0

# zero-shot
python test.py -d JDDC -m roberta -s base -t 2 -z
python test.py -d JDDC -m roberta -s large -t 2 -z -b 4

# fine-tuning
python test.py -d JDDC -m roberta -s base -t 2 -b 8
python test.py -d JDDC -m roberta -s large -t 2 -b 2
