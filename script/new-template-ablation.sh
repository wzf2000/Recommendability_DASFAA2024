export CUDA_VISIBLE_DEVICES=

# zero-shot
python test.py -l zh -d DuRecDial -m bert -s base -t 1 -z -e 20 --new
python test.py -l en -d DuRecDial -m bert -s base -t 1 -z -e 20 --new

# fine-tuning
python test.py -l zh -d DuRecDial -m bert -s base -t 1 -b 8 -e 20 --new
python test.py -l en -d DuRecDial -m bert -s base -t 1 -b 8 -e 20 --new
