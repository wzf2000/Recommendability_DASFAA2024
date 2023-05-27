export CUDA_VISIBLE_DEVICES=0

# zero-shot
python test.py -d JDDC -m gpt2 -s base -t 2 -z
python test.py -d JDDC -m gpt2 -s xl -t 2 -z -b 1

# fine-tuning
python test.py -d JDDC -m gpt2 -s base -t 2 -b 8
