# Specialization after Generalization: Towards Understanding Test-Time Training in Foundation Models

[![Paper](https://img.shields.io/badge/arXiv-2509.24510-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2509.24510)
[![Model](https://img.shields.io/badge/HuggingFace-SAE-orange.svg?logo=huggingface)](https://huggingface.co/patrikwolf/clip-topk-sae)
<!-- [![Code](https://img.shields.io/badge/GitHub-ttt__theory-blue.svg?logo=github)](https://github.com/patrikwolf/ttt_theory) -->

This repository supports our paper, which investigates **why and when** Test-Time Training (TTT) improves foundation 
models, even on in-distribution data. Our work provides both **theoretical and empirical evidence** for the mechanisms
behind TTT, and identifies the regimes in which it is most effective.

Please cite our work if you use this code, in full or in part, for your research (see [BibTeX below](#citation)):

- [Specialization after Generalization: Towards Understanding Test-Time Training in Foundation Models](https://arxiv.org/abs/2509.24510)

## 📈 Empirical Validation

We validate the model’s key assumptions through experiments with sparse autoencoders on ImageNet and scaling studies on 
MNIST and CLIP:
- **Similarity Preservation:** Learned features preserve the concept space’s similarity structure. 
- **Local Sparsity:** Ground truth functions are well-approximated by sparse linear combinations of concepts within 
local input neighborhoods.
- **Implicit Concept Selection:** TTT selectively tunes only the few concepts relevant for each test task.
- **Scaling Results:** TTT yields notable accuracy gains in underparameterized regimes, but gains taper off as model 
size increases.
These findings are crucial for practitioners seeking to determine when TTT should be applied in real-world applications.

## 💡 Overview

The codebase is organized into two main components reflecting our paper's structure:

- `sae/`: Sparse autoencoder experiments from Section 3 for validating the linear representation hypothesis, including 
top-kSAEs and utilities for concept sparsity studies.
- `scaling/`: Implements scaling studies from Section 4 for image tasks, comparing global models and TTT variants under 
different model/dataset sizes. Code covers MNIST and ImageNet experiments, including global classifiers, linear heads, 
TTT procedures, and majority voting.

## 🚀 Quickstart

Create a Python environment (Python 3.10+), for example, by running:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

which creates a virtual environment. Then install dependencies:

```bash
pip install -e . 
```

A description of how to run the experiments is provided in the `README.md` files of the respective folders (`sae/` and `scaling/`).

## 🗂️ Repository Structure

- `sae/`
  - `sae_topk.py`: Top-k sparse autoencoder implementation.
  - `extract_clip_embd.py`: Script to extract and normalize 512-d CLIP \<CLS\> embeddings consistently.
  - `clip_sae_checkpoint.pt`: Trained SAE checkpoint including `config` and `model_state_dict`.
  - `config.json`: Training schedule and hyperparameters used for the provided checkpoint.
- `scaling/`
  - `mnist/`: MNIST classification and TTT variants.
    - Global classifier: `a_global_classifier/train_global_model.py`
    - Linear head: `b_linear_head/train_mnist_linear_head.py`
    - TTT linear head: `c_ttt_linear_head/ttt_linear.py`
    - Scaled CNNs: `d_scale_model/train_mnist_scaled_models.py`
    - LFT-style/TTT CNN: `e_ttt_scaled_cnn/mnist_lft_cnn.py`
    - Majority-vote CNN: `f_majority_scaled_cnn/mnist_majority_vote_cnn.py`
    - Dataset scaling: `g_scale_dataset/mnist_scale_dataset.py`
    - Global eval: `h_lft_global_eval/lft_global_eval.py`
    - Mixture-of-Experts: `i_mixture_of_experts/mnist_moe.py`
    - Parameters: `parameters/` contains JSON/YAML configs used by the scripts.
  - `imagenet/`: ImageNet/CLIP linear and MLP heads plus scaling.
    - Global linear: `a_global_linear/train_global_linear_model.py`
    - TTT linear head: `b_ttt_linear_head/ttt_linear_head.py`
    - Global MLP head: `c_mlp_heads/train_global_mlp_head.py`
    - TTT MLP head: `d_ttt_mlp_heads/ttt_mlp_head.py`
    - Majority vote: `e_majority/majority_vote_{linear,mlp}.py`
    - Dataset scaling: `f_scale_dataset/eval_scale_dataset.py`
    - Global eval: `g_ttt_global_eval/ttt_global_eval.py`
    - Parameters: `parameters/` includes optimal params, helpers, and references.
  - `log_book/`: Minimal logging helpers (`read_and_write.py`).

## Citation

If you use this repository in your research, please cite the accompanying work:

```bibtex
@misc{hübotter2025specializationgeneralizationunderstandingtesttime,
      title={Specialization after Generalization: Towards Understanding Test-Time Training in Foundation Models}, 
      author={Jonas Hübotter and Patrik Wolf and Alexander Shevchenko and Dennis Jüni and Andreas Krause and Gil Kur},
      year={2025},
      eprint={2509.24510},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.24510}, 
}
```