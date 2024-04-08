export GPU=2

# roberta
config=config/durecdial-roberta-zh.yaml bash scripts/multi-seed.sh
config=config/durecdial-roberta-en.yaml bash scripts/multi-seed.sh

# deberta
config=config/durecdial-deberta-zh.yaml bash scripts/multi-seed.sh
config=config/durecdial-deberta-en.yaml bash scripts/multi-seed.sh

# bert
config=config/durecdial-bert-zh.yaml bash scripts/multi-seed.sh
config=config/durecdial-bert-en.yaml bash scripts/multi-seed.sh
