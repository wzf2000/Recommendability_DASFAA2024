# Recommendability_DASFAA2024

Source codes for paper "To Recommend or Not: Recommendability Identification in Conversations with Pre-trained Language Models" at DASFAA 2024

This branch is the implementation of the *Soft Prompt Tuning* methods of the paper.

We use the [P-tuning-V2](https://github.com/THUDM/P-tuning-v2) repository as the code base and modify it to fit our task.

## Setup

We conduct our experiment with Anaconda3. If you have installed Anaconda3, then create the environment for P-tuning v2:

```shell
conda create -n pt2 python=3.8.5
conda activate pt2
```

After we setup basic conda environment, install pytorch related packages via:

```shell
conda install -n pt2 pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
```

Finally, install other python packages we need:

```shell
pip install -r requirements.txt
```

## Data

Download the DuRecDial and JDDCRec datasets from the following links:
- [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/14387eba870147a084ca/)
- [Google Drive](https://drive.google.com/drive/folders/1XlZmimNGEaCHIMqAP2vBUJ9bAj5M8sqH?usp=sharing)

After downloading the datasets, unzip them and put them in the `../datasets` folder, i.e., the folder structure should be like this:
```
../datasets/
    DuRecDial/
        *.json
    JDDC/
        *.json
```

> You can also change the dataset path setting in `tasks/rec/dataset.py` to fit your own path.

## Run the code

We give examples of running the code on both DuRecDial and JDDCRec datasets. You can check the scripts in the `run_script` folder, e.g., `run_script/run_durecdial_f1.sh`, `run_script/run_jddc_multi_seed.sh`, etc.

## Citation
If you find our work useful, please do not save your star and cite our work:
```
@article{wang2024recommend,
  title={To Recommend or Not: Recommendability Identification in Conversations with Pre-trained Language Models},
  author={Wang, Zhefan and Ma, Weizhi and Zhang, Min},
  journal={arXiv preprint arXiv:2403.18628},
  year={2024}
}
```

And if the P-tuning repository is helpful, please also cite the original work:
```
@article{DBLP:journals/corr/abs-2110-07602,
  author    = {Xiao Liu and
               Kaixuan Ji and
               Yicheng Fu and
               Zhengxiao Du and
               Zhilin Yang and
               Jie Tang},
  title     = {P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally
               Across Scales and Tasks},
  journal   = {CoRR},
  volume    = {abs/2110.07602},
  year      = {2021},
  url       = {https://arxiv.org/abs/2110.07602},
  eprinttype = {arXiv},
  eprint    = {2110.07602},
  timestamp = {Fri, 22 Oct 2021 13:33:09 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2110-07602.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
