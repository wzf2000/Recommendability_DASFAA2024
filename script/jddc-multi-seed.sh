export CUDA_VISIBLE_DEVICES=
export template=2
export dataset=JDDC

# zh
export language=zh
model=bert size=base bs=8 suffix= bash script/multi-seed.sh
model=roberta size=base bs=8 suffix= bash script/multi-seed.sh
model=roberta size=large bs=2 suffix= bash script/multi-seed.sh
