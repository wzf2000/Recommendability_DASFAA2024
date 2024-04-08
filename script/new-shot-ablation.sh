export CUDA_VISIBLE_DEVICES=

# unbalance

## zh
python test.py -l zh -d DuRecDial -m bert -s base -t 0 -b 8 -f --times 100 --few_shot_num 30 -e 20 --new
python test.py -l zh -d DuRecDial -m bert -s base -t 0 -b 8 -f --times 10 --few_shot_num 300 -e 20 --new
python test.py -l zh -d DuRecDial -m bert -s base -t 0 -b 8 -f --few_shot_num 3000 -e 20 --new

## en
python test.py -l en -d DuRecDial -m bert -s base -t 0 -b 8 -f --times 100 --few_shot_num 30 -e 20 --new
python test.py -l en -d DuRecDial -m bert -s base -t 0 -b 8 -f --times 10 --few_shot_num 300 -e 20 --new
python test.py -l en -d DuRecDial -m bert -s base -t 0 -b 8 -f --few_shot_num 3000 -e 20 --new

# balance

## zh
python test.py -l zh -d DuRecDial -m bert -s base -t 0 -b 8 -f --few_shot_num 30 --balance --times 100 -e 20 --new
python test.py -l zh -d DuRecDial -m bert -s base -t 0 -b 8 -f --few_shot_num 300 --balance --times 10 -e 20 --new

## en
python test.py -l en -d DuRecDial -m bert -s base -t 0 -b 8 -f --few_shot_num 30 --balance --times 100 -e 20 --new
python test.py -l en -d DuRecDial -m bert -s base -t 0 -b 8 -f --few_shot_num 300 --balance --times 10 -e 20 --new
