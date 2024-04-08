export CUDA_VISIBLE_DEVICES=

# zh

## bert
python test.py -l zh -d DuRecDial -m bert -s base -t 0 -b 8 -e 20 --new

## roberta
python test.py -l zh -d DuRecDial -m roberta -s base -t 0 -b 8 -e 20 --new
python test.py -l zh -d DuRecDial -m roberta -s large -t 0 -b 2 -e 20 --new

# en

## bert
python test.py -l en -d DuRecDial -m bert -s base -t 0 -b 8 -e 20 --new
python test.py -l en -d DuRecDial -m bert -s large -t 0 -b 2 -e 20 --new -w 0.002

## roberta
python test.py -l en -d DuRecDial -m roberta -s base -t 0 -b 8 -e 20 --new
python test.py -l en -d DuRecDial -m roberta -s large -t 0 -b 2 -e 20 --new -w 0.002

## gpt2
python test.py -l en -d DuRecDial -m gpt2 -s base -t 0 -b 8 -e 20 --new
python test.py -l en -d DuRecDial -m gpt2 -s medium -t 0 -b 2 -e 20 --new
