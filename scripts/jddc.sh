export GPU=0

config=config/jddc-roberta.yaml bash scripts/multi-seed.sh
config=config/jddc-deberta.yaml bash scripts/multi-seed.sh
config=config/jddc-bert.yaml bash scripts/multi-seed.sh
