# DexAnyTwist: Learning General Dexterous Twisting with Hybrid Manipulation System Identification

> **Note:** This paper is currently under review. The full code and assets will be released upon acceptance.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project Page](https://img.shields.io/badge/Project-Website-blue)](https://prevalenter.github.io/dexanytwist.github.io/)

**DexAnyTwist** is a general reinforcement learning framework for dexterous in-hand twisting manipulation. It addresses the challenge of hybrid manipulation dynamics by introducing a dynamic subsystem sample identification strategy. The framework is trained on a large-scale dataset of over 300 objects and achieves robust zero-shot sim-to-real transfer on diverse everyday objects using the GX10 dexterous hand.

![Teaser](imgs/head.jpg)


## 🛠️ Installation

This project is built on **Isaac Gym** and **PyTorch**.

### Prerequisites
* Python 3.8+
* PyTorch 1.10+
* NVIDIA Driver & CUDA

### 1. Environment Setup
```bash
conda create -n dexanytwist python=3.8
conda activate dexanytwist
pip install torch torchvision torchaudio --extra-index-url [https://download.pytorch.org/whl/cu113](https://download.pytorch.org/whl/cu113)
