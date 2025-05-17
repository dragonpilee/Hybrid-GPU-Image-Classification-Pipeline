import time
import torch
import cupy as cp
import numpy as np
from numba import cuda
from fastai.vision.all import *

print("🚀 Starting Hybrid GPU Image Classification Pipeline")

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load dataset (CIFAR-10) with fastai DataLoaders
batch_size = 128
dls = ImageDataLoaders.from_folder(
    untar_data(URLs.CIFAR),
    train='train',
    valid='test',
    bs=batch_size,
    item_tfms=Resize(32),
    device=device
)
print(f"Dataset loaded with batch size {batch_size}")

# CuPy normalization function
def gpu_normalize_images(images):
    # images shape (bs, ch, h, w), float32 assumed in range [0,1]
    images_cp = cp.asarray(images)
    mean = cp.array([0.4914, 0.4822, 0.4465], dtype=cp.float32).reshape(3,1,1)
    std = cp.array([0.247, 0.243, 0.261], dtype=cp.float32).reshape(3,1,1)
    normalized = (images_cp - mean) / std
    return cp.asnumpy(normalized)

# Numba GPU kernel for brightness augmentation
@cuda.jit
def brightness_kernel(images, brightness_factor):
    idx = cuda.grid(1)
    size = images.size
    if idx < size:
        images[idx] = min(images[idx] * brightness_factor, 1.0)

def gpu_brightness_augment(images):
    # images shape (bs, ch, h, w)
    images_cp = cp.asarray(images)
    brightness_factor = 1.2  # Increase brightness by 20%
    flat = images_cp.ravel()
    threads_per_block = 256
    blocks = (flat.size + threads_per_block - 1) // threads_per_block
    brightness_kernel[blocks, threads_per_block](flat, brightness_factor)
    return cp.asnumpy(images_cp)

# Preprocess batch with GPU acceleration steps
def preprocess_batch(batch):
    imgs, labels = batch
    # Move images to CPU and convert to numpy
    imgs_np = imgs.cpu().numpy()
    # Normalize on GPU with CuPy
    imgs_np = gpu_normalize_images(imgs_np)
    # Brightness augmentation on GPU with Numba
    imgs_np = gpu_brightness_augment(imgs_np)
    # Convert back to tensor and send to device
    imgs_tensor = torch.tensor(imgs_np).float().to(device)
    return imgs_tensor, labels

# Define a simple CNN model with FastAI
def get_model():
    learn = vision_learner(dls, resnet18, metrics=accuracy)
    return learn

# Train the model with preprocessing
def train_model():
    learn = get_model()
    print("Preprocessing sample batch with GPU kernels...")
    batch = dls.one_batch()
    x, y = preprocess_batch(batch)  # Preprocess a sample batch to check
    print("Starting training...")
    start = time.time()
    learn.fit_one_cycle(1)
    end = time.time()
    print(f"✅ FastAI training completed\n⏱️ Time: {end - start:.2f} sec")

if __name__ == '__main__':
    train_model()
