export CUDA_VISIBLE_DEVICES=9

# zero-shot
python test.py -l zh -d DuRecDial -m bert -s base -t 1 -z
python test.py -l en -d DuRecDial -m bert -s base -t 1 -z

# fine-tuning
python test.py -l zh -d DuRecDial -m bert -s base -t 1 -b 8
python test.py -l en -d DuRecDial -m bert -s base -t 1 -b 8
