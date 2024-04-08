# Recommendability_DASFAA2024

Source codes for paper "To Recommend or Not: Recommendability Identification in Conversations with Pre-trained Language Models" at DASFAA 2024

This branch is the implementation of the *Hard Prompt Tuning* and *Zero-shot Prompt Evaluation* methods of the paper.

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