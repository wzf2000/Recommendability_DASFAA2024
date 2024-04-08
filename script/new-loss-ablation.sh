export CUDA_VISIBLE_DEVICES=

# wce
python test.py -l zh -d DuRecDial -m bert -s base -t 0 -b 8 --loss wce --alpha 0.75 -e 20 --new
python test.py -l zh -d DuRecDial -m bert -s base -t 0 -b 8 --loss wce --alpha 0.5 -e 20 --new
python test.py -l zh -d DuRecDial -m bert -s base -t 0 -b 8 --loss wce --alpha 0.25 -e 20 --new

# focal
python test.py -l zh -d DuRecDial -m bert -s base -t 0 -b 8 --loss focal --alpha 1 --gamma 1 -e 20 --new
python test.py -l zh -d DuRecDial -m bert -s base -t 0 -b 8 --loss focal --alpha 1 --gamma 2 -e 20 --new

# dsc
python test.py -l zh -d DuRecDial -m bert -s base -t 0 -b 8 --loss dsc --gamma 0.02 --smooth 1 --dice_square -e 20 --new
python test.py -l zh -d DuRecDial -m bert -s base -t 0 -b 8 --loss dsc --gamma 0.04 --smooth 1 --dice_square -e 20 --new
