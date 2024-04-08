# Recommendability_DASFAA2024
Source codes for paper "To Recommend or Not: Recommendability Identification in Conversations with Pre-trained Language Models" at DASFAA 2024

We provide the code for different methods mentioned in the paper in different branches:
- For *baseline* models, please refer to the branch `baseline`. You can checkout to the branch by running `git checkout baseline`.
- For *Hard Prompt Learning* or *Zero-shot Prompt Evaluation* methods, please refer to the branch `prompt-tuning` or the `main` branch. You can checkout to the branch by running `git checkout prompt-tuning`.
- For *Soft Prompt Tuning* methods, please refer to the branch `P-tuning`. You can checkout to the branch by running `git checkout P-tuning`.

Note that the code in the `main` branch is the same as the code in the `prompt-tuning` branch and it just contains the code for the *Hard Prompt Learning* and *Zero-shot Prompt Evaluation* methods.

## Requirements

0. Make sure the python version is greater than or equal to 3.8.16. We do not test the code on other versions.

1. Run the following commands to install PyTorch (Note: change the URL setting if using another version of CUDA):
    ```shell
    pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
    ```
2. Run the following commands to install dependencies:
    ```shell
    pip install -r requirements.txt
    ```

## Run the code

We give examples of running the code on both DuRecDial and JDDCRec datasets. You can check the scripts in the `script` folder.

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

And if the `OpenPrompt` library is helpful, please also cite the following paper:
```
@article{ding2021openprompt,
  title={OpenPrompt: An Open-source Framework for Prompt-learning},
  author={Ding, Ning and Hu, Shengding and Zhao, Weilin and Chen, Yulin and Liu, Zhiyuan and Zheng, Hai-Tao and Sun, Maosong},
  journal={arXiv preprint arXiv:2111.01998},
  year={2021}
}
```
