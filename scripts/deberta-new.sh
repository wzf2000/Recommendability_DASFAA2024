export GPU=0
export config=config/jddc-deberta.yaml

python test.py --gpu $GPU --restore --config $config --seed 2019
python test.py --gpu $GPU --restore --config $config --seed 2020
python test.py --gpu $GPU --restore --config $config --seed 2021
python test.py --gpu $GPU --restore --config $config --seed 2022
python test.py --gpu $GPU --restore --config $config --seed 2023

export config=config/durecdial-deberta-zh.yaml

python test.py --gpu $GPU --restore --config $config --seed 2023
