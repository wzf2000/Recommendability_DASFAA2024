export CUDA_VISIBLE_DEVICES=0

# zero-shot
python test.py -d DuRecDial -m gpt2 -s base -t 0 -z
python test.py -d DuRecDial -m gpt2 -s xl -t 0 -z -b 1

# fine-tuning
python test.py -d DuRecDial -m gpt2 -s base -t 0 -b 8
