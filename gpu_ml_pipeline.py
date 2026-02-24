import time
import torch
import cupy as cp
import numpy as np
from numba import cuda
from fastai.vision.all import *
from torch.utils.dlpack import to_dlpack, from_dlpack

print("🚀 Starting Optimized Hybrid GPU Image Classification Pipeline")

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- GPU KERNELS & UTILS ---

def torch_to_cupy(t):
    return cp.from_dlpack(to_dlpack(t))

def cupy_to_torch(c):
    return from_dlpack(c.toDlpack())

@cuda.jit
def brightness_kernel(images, brightness_factor):
    idx = cuda.grid(1)
    if idx < images.size:
        images[idx] = min(images[idx] * brightness_factor, 1.0)

@cuda.jit
def salt_pepper_kernel(images, prob, seed):
    idx = cuda.grid(1)
    if idx < images.size:
        # Simple pseudo-random logic for noise
        res = (idx * 1103515245 + 12345 + seed) & 0x7fffffff
        random_val = res / 0x7fffffff
        if random_val < prob / 2:
            images[idx] = 0.0
        elif random_val < prob:
            images[idx] = 1.0

class GPUPipelineTransform(Transform):
    def __init__(self, brightness=1.2, noise_prob=0.02):
        self.brightness = brightness
        self.noise_prob = noise_prob
        self.mean = cp.array([0.4914, 0.4822, 0.4465], dtype=cp.float32).reshape(1,3,1,1)
        self.std = cp.array([0.247, 0.243, 0.261], dtype=cp.float32).reshape(1,3,1,1)

    def encodes(self, b):
        if not isinstance(b, tuple) or len(b) != 2: return b
        imgs, labels = b
        if not imgs.is_cuda: return b

        # Zero-copy share with CuPy
        imgs_cp = torch_to_cupy(imgs)

        # 1. Normalization
        imgs_cp = (imgs_cp - self.mean) / self.std

        # 2. Brightness Kernel
        flat = imgs_cp.ravel()
        threads = 256
        blocks = (flat.size + threads - 1) // threads
        brightness_kernel[blocks, threads](flat, self.brightness)

        # 3. Noise Kernel
        salt_pepper_kernel[blocks, threads](flat, self.noise_prob, int(time.time()))

        return cupy_to_torch(imgs_cp), labels

# --- MAIN SETUP ---

# Load dataset (CIFAR-10)
path = untar_data(URLs.CIFAR)
dls = ImageDataLoaders.from_folder(
    path, train='train', valid='test', 
    bs=128, item_tfms=Resize(32),
    batch_tfms=[IntToFloatTensor(), GPUPipelineTransform()],
    device=device
)

def train_model():
    learn = vision_learner(dls, resnet18, metrics=accuracy).to_fp16()
    print("Starting optimized training (Mixed Precision + Zero-Copy GPU Preprocessing)...")
    
    start = time.time()
    # Using small number of epochs for demo
    learn.fit_one_cycle(1, 1e-3)
    end = time.time()
    
    print(f"✅ Training completed in {end - start:.2f} sec")

if __name__ == '__main__':
    train_model()
