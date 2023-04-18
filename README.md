# Difformer

The offical codebase for [Difformer: Empowering Diffusion Models on the Embedding Space for Text Generation](https://arxiv.org/abs/2212.09412).

## Getting started

Our implementation is based on Python 3.8, PyTorch 1.11 and Fairseq 0.10.2. The following command will install the dependencies and this package in a Conda environment:

```shell
conda install pytorch==1.11.0 -c pytorch
pip install -e .
```

## Data preparing

We follow the [instructions of Fairseq](https://github.com/facebookresearch/fairseq/tree/main/examples/translation#iwslt14-german-to-english-transformer) to preprocess the translation datasets. Then we adopt knowledge distillation using Transformer models trained on the same datasets. To binarize the distilled and tokenized datasets, run following command (take the IWSLT14 De-En dataset as an example):

```shell
fairseq-preprocess \
    --source-lang de --target-lang en \
    --trainpref {PATH-TO-YOUR-DATASET}/train \
    --validpref {PATH-TO-YOUR-DATASET}/valid \
    --testpref {PATH-TO-YOUR-DATASET}/test \
    --destdir data-bin/iwslt14_de_en_distill \
    --joined-dictionary \
    --workers 20
```

## Training

All training and evaluation scripts are put in the `./scripts` directory. For example, to train Difformer on the IWSLT14 De-En dataset, simply run:

```shell
bash scripts/iwslt14_de_en/train.sh
```

## Decoding and evaluation

We apply checkpoint averaging instead of exponential moving averaging (EMA) of parameters before evaluation. Specifically, we report the performance of averaging best 5 checkpoints. An example script is provided:

```shell
bash scripts/iwslt14_de_en/evaluate.sh
```

The model checkpoints and data-bins are provided [here](https://drive.google.com/drive/folders/1XpF6sPd7EPcnz9uEA8-dplRvOAqeUlx1), with performance listed below.

|    Dataset    | BLEU  | SacreBLEU | COMET  | ROUGE-L | BERTScore |
| :-----------: | :---: | :-------: | :----: | :-----: | :-------: |
| IWSLT14 De-En | 34.48 |   33.5    | 0.7875 |    -    |     -     |
|  WMT14 En-De  | 27.74 |   26.2    | 0.8257 |    -    |     -     |
|      QQP      | 30.43 |     -     |   -    |  61.25  |   85.02   |
|   Wiki-Auto   | 40.77 |     -     |   -    |  59.86  |   82.22   |

Here are implementations we used in evaluation: [SacreBLEU](https://github.com/mjpost/sacrebleu), [COMET](https://github.com/Unbabel/COMET), [ROUGE](https://github.com/pltrdy/files2rouge) and [BERTScore](https://github.com/Tiiiger/bert_score).

## Citation

Please cite our paper if you find this codebase useful:

```bibtex
@article{gao2022difformer,
  title={Difformer: Empowering Diffusion Model on Embedding Space for Text Generation},
  author={Gao, Zhujin and Guo, Junliang and Tan, Xu and Zhu, Yongxin and Zhang, Fang and Bian, Jiang and Xu, Linli},
  journal={arXiv preprint arXiv:2212.09412},
  year={2022}
}
```

## Acknowledgments

This codebase includes the code of [improved-diffusion](https://github.com/openai/improved-diffusion). Thank them for their contributions to the community.
