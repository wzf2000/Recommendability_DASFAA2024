# Recommendability_DASFAA2024

Source codes for paper "To Recommend or Not: Recommendability Identification in Conversations with Pre-trained Language Models" at DASFAA 2024

This branch is the implementation of the *baseline* methods of the paper, i.e., *ConvBERT*, *ConvDeBERTa*, *ConvRoBERTa*.

## Setup

0. Make sure the python version is equal to 3.6.13. We do not test the code on other versions.

1. Install the requirements for `crslab` according to the instructions in this [repo](https://github.com/RUCAIBox/CRSLab).
2. Run the following commands to install other dependencies:
    ```shell
    pip install -r requirements.txt
    ```

## Run the code

We give examples of running the code on both DuRecDial and JDDCRec datasets. You can check the scripts in the `scripts` folder.

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

And if the `CRSLab` is useful for your research, please also cite the following paper:
```
@article{crslab,
    title={CRSLab: An Open-Source Toolkit for Building Conversational Recommender System},
    author={Kun Zhou, Xiaolei Wang, Yuanhang Zhou, Chenzhan Shang, Yuan Cheng, Wayne Xin Zhao, Yaliang Li, Ji-Rong Wen},
    year={2021},
    journal={arXiv preprint arXiv:2101.00939}
}
```
