export CUDA_VISIBLE_DEVICES=7

# fine-tuning
python test.py -d DuRecDial -m glm -s base -t 0 -b 1
python test.py -l en -d DuRecDial -m glm -s base -t 0 -b 1
python test.py -d JDDC -m glm -s base -t 2 -b 1