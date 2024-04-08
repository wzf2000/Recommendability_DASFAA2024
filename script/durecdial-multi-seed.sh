export CUDA_VISIBLE_DEVICES=
export template=0
export dataset=DuRecDial

# zh
export language=zh
model=bert size=base bs=8 suffix= bash script/multi-seed.sh
model=roberta size=base bs=8 suffix= bash script/multi-seed.sh
model=roberta size=large bs=2 suffix= bash script/multi-seed.sh

# en
export language=en
model=bert size=base bs=8 suffix= bash script/multi-seed.sh
model=bert size=large bs=2 suffix= bash script/multi-seed.sh
model=roberta size=base bs=8 suffix= bash script/multi-seed.sh
model=roberta size=large bs=2 suffix= bash script/multi-seed.sh
model=gpt2 size=base bs=8 suffix= bash script/multi-seed.sh
model=gpt2 size=medium bs=2 suffix= bash script/multi-seed.sh
