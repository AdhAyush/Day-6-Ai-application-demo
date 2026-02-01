# download_samples.py
from torchvision import datasets
from PIL import Image
import os

# Download MNIST
mnist = datasets.MNIST('./data', train=False, download=True)

# Save first 10 test images
os.makedirs('static/sample_images', exist_ok=True)
for i in range(10):
    img, label = mnist[i]
    img.save(f'static/sample_images/digit_{label}.png')

print("✅ Saved 10 sample images!")