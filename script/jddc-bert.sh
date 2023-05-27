export CUDA_VISIBLE_DEVICES=9

# zero-shot
python test.py -d JDDC -m bert -s base -t 2 -z

# fine-tuning
python test.py -d JDDC -m bert -s base -t 2 -b 8
