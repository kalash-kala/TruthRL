<h1 align="center">
TruthRL 
</h1>

<h3 align="center">
Incentivizing Truthful LLMs via Reinforcement Learning <br>
[<a href="https://arxiv.org/abs/2509.25760">arXiv</a>]
[<a href="https://x.com/weizhepei/status/1973211813522317519">Summary</a>]
</h3>

TruthRL is a simple yet effective truthfulness-driven reinforcement learning (RL) method that significantly reduces hallucinations in large language models (LLMs) by enabling proper abstention while preserving accuracy.


## Why TruthRL?
***Factual accuracy alone does NOT necessarily guarantee truthfulness!***

A model that answers fewer questions correctly while reliably abstaining when uncertain is far more trustworthy than a higher-accuracy model that frequently fabricates plausible but incorrect answers.
<center>
  <img width="1084" height="409" alt="image" src="https://github.com/facebookresearch/TruthRL/blob/main/TruthRL.png" />
</center>

In vanilla supervised fine-tuning (SFT) or RL, the model is optimized solely for accuracy, implicitly rewarding hallucinations over abstentions and thus always attempting to answer or guess, which ultimately compromises truthfulness. In contrast, TruthRL not only **rewards correct answers**, but explicitly **penalizes hallucinations**, and **treats abstentions neutrally**, thereby leading to greater truthfulness.

## Installation
Run the following script to create a Python virtual environment for TruthRL training.
```bash
conda create -n truthrl-verl python=3.10 -y
conda activate truthrl-verl
conda install nvidia/label/cuda-12.4.0::cuda-toolkit

cd training/verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh

pip install numpy==1.26.1 opentelemetry-sdk==1.26.0 opentelemetry-sdk==1.26.0 click==8.2.1 tensordict==0.8.1

pip install --no-deps -e .

huggingface-cli login
wandb login
```


## Training

The training requires an LLM verifier to judge whether the predicted answer aligns with the reference answer and produce reward signals. If not hosting locally, please change `OPENAI_API_BASE` in `train_grpo.sh` to the base URL where you host the verifier model. By default, the training script is set for 8 x H100 80G GPUs. Please adjust `N_GPUS`  based on your compute resource.

```shell
conda activate truthrl-verl
bash train_grpo.sh
```

## Evaluation

Run the following script to create a Python virtual environment for TruthRL training.
```bash
conda create -n truthrl-eval python=3.10 -y
conda activate truthrl-eval

cd evaluation
pip install -r requirements.txt
```

Use the following script to evaluate the model. Note that the evaluation also requires a LLM to judge whether the predicted answer aligns with the reference answer. If not hosting locally, please change `api_url` in `evaluate.py` to the base URL where you host the verifier model.


```shell
conda activate truthrl-eval
python evaluate.py
```


## Bugs or Questions?
If you have any questions related to the code or the paper, feel free to email Zhepei (zhepei.wei@virginia.edu). If you encounter any problems when using the code, or want to report a bug, feel free to open an issue! Please try to specify the problem with details so we can help you better and quicker!

## Citation
Please cite our paper if you find the repo helpful in your work:

```bibtex
@article{
wei2025truthrl,
title={Truth{RL}: Incentivizing Truthful {LLMs} via Reinforcement Learning},
author={Wei, Zhepei and Yang, Xiao and Sun, Kai and Wang, Jiaqi and Shao, Rulin and Chen, Sean and Kachuee, Mohammad and Gollapudi, Teja and Liao, Tony and Scheffer, Nicolas and Wanga, Rakesh and Kumar, Anuj and Meng, Yu and Yih, Wen-tau and Dong, Xin Luna},
journal={arXiv preprint arXiv:2509.25760},
year={2025},
}
```

## License
TruthRL is Creative Commons Attribution-NonCommercial 4.0 International License licensed, as found in the LICENSE file.

