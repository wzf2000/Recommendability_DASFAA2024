export CUDA_VISIBLE_DEVICES=

python test.py -l zh -d JDDC -m bert -s base -t 2 -b 8 -z
python test.py -l zh -d JDDC -m roberta -s base -t 2 -b 8 -z
python test.py -l zh -d JDDC -m roberta -s large -t 2 -b 2 -z
python test.py -l zh -d JDDC -m glm -s base -t 2 -b 1 -z
python test.py -l zh -d JDDC -m glm2 -s base -t 2 -b 1 -z