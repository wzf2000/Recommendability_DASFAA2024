export CUDA_VISIBLE_DEVICES=0

# zero-shot
python test.py -l en -d DuRecDial -m gpt2 -s base -t 0 -z
python test.py -l en -d DuRecDial -m gpt2 -s medium -t 0 -z
python test.py -l en -d DuRecDial -m gpt2 -s large -t 0 -z
python test.py -l en -d DuRecDial -m gpt2 -s xl -t 0 -z

# fine-tuning
python test.py -l en -d DuRecDial -m gpt2 -s base -t 0
python test.py -l en -d DuRecDial -m gpt2 -s medium -t 0 -b 8
