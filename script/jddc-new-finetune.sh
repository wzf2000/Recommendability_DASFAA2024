export CUDA_VISIBLE_DEVICES=0

# zh

## bert
python test.py -l zh -d JDDC -m bert -s base -t 2 -b 8 -e 20 --new

## roberta
python test.py -l zh -d JDDC -m roberta -s base -t 2 -b 8 -e 20 --new
python test.py -l zh -d JDDC -m roberta -s large -t 2 -b 2 -e 20 --new