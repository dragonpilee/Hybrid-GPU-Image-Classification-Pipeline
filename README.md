
# 🚀 Hybrid GPU Image Classification Pipeline  
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAI](https://img.shields.io/badge/FastAI-181717?style=for-the-badge&logo=fastapi&logoColor=white)
![CUDA](https://img.shields.io/badge/NVIDIA-CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

> ⚡ A high-performance deep learning pipeline combining PyTorch, FastAI, CuPy, and Numba — **optimized for NVIDIA RTX GPUs** — to preprocess and train on the CIFAR-10 dataset with maximum GPU acceleration.

---

## 🧠 Overview

This repository demonstrates a **hybrid GPU pipeline** that offloads **image preprocessing** and **augmentation** to the GPU using:

- 🧮 **CuPy**: Fast NumPy-like GPU array computations  
- 🔬 **Numba**: Custom CUDA kernels for image brightness enhancement  
- 🐍 **FastAI**: Rapid model training using PyTorch under the hood  
- 🎮 **RTX GPU Optimized**: Designed to fully leverage your RTX GPU (20xx, 30xx, or 40xx series)

---

## 📊 Demo Output

```bash
🚀 Starting Hybrid GPU Image Classification Pipeline
Using device: cuda
Dataset loaded with batch size 128
Preprocessing sample batch with GPU kernels...
Starting training...
✅ FastAI training completed
⏱️ Time: 18.53 sec
```

---

## 🏗️ Pipeline Architecture

```
┌────────────────────┐
│ CIFAR-10 Dataset   │
└─────────┬──────────┘
          ↓
   FastAI DataLoaders
          ↓
  ┌────────────────────────────┐
  │ GPU Preprocessing Steps    │
  ├────────────────────────────┤
  │ 1. CuPy Normalization       │
  │ 2. Numba Brightness Kernel  │
  └────────────────────────────┘
          ↓
   FastAI ResNet-18 Model
          ↓
       Training Loop
```

---

## ⚙️ Installation

### ✅ Prerequisites

- NVIDIA **RTX GPU** with CUDA support  
- Python 3.8+  
- CUDA drivers installed and working  
- CUDA-compatible versions of PyTorch and CuPy

### 📦 Install Dependencies

```bash
# For CUDA 11.8 and RTX GPUs
pip install cupy-cuda118

# Core packages
pip install torch torchvision fastai numba
```

> 💡 Choose the CuPy version that matches your installed CUDA toolkit:  
> See the [CuPy install matrix](https://docs.cupy.dev/en/stable/install.html#using-pip).

---

## 🧪 How to Run

```bash
https://github.com/dragonpilee/Hybrid-GPU-Image-Classification-Pipeline.git
cd hybrid-gpu-classifier
python train.py
```

---

## 🔍 Code Highlights

### 🔧 `gpu_normalize_images()`

Normalizes RGB image tensors using CuPy directly on GPU memory.

### 💡 `brightness_kernel` (Numba)

Custom CUDA kernel that increases brightness pixel-wise on the GPU.

### 🧼 `preprocess_batch()`

Combines GPU normalization and augmentation, then converts back to PyTorch tensors.

### 🧠 `train_model()`

Initializes FastAI’s ResNet-18 model and performs training with one-cycle policy.

---

## ⚡ Performance Tips

| Setting                     | Recommendation                     |
|----------------------------|-------------------------------------|
| GPU                        | NVIDIA RTX 2060/3060/4090 or higher |
| Batch Size                 | 128–256 for optimal GPU utilization |
| Data Preprocessing         | CuPy + Numba (already integrated)   |
| Precision                  | Add mixed precision for even faster training |
| Memory Management          | Use `.half()` and `torch.cuda.amp` for FP16 |

---

## 🚀 Future Improvements

- 🔁 Integrate GPU-accelerated preprocessing inside FastAI’s transform pipeline  
- 🎨 Add more augmentation types (contrast, noise, rotation) via custom CUDA kernels  
- 📈 Benchmark performance across different GPUs (RTX 2060 vs 4090)  
- 💾 Add model saving, evaluation, and inference scripts  

---

## 📜 License

This project is licensed under the **MIT License**.  
Feel free to fork, modify, and use it in your own projects.

---

## 👨‍💻 Author

**Alan Cyril Sunny**  
📧 alan_cyril@yahoo.com  
🐙 [GitHub](https://github.com/dragonpilee)

---

## 🌟 Show Your Support

If you found this useful, consider starring ⭐ the repo or sharing it with others!
