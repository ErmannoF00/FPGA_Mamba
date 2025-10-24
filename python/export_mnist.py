import torch
from torchvision import datasets, transforms
import numpy as np

dataset = datasets.MNIST('.', download=True, train=False, transform=transforms.ToTensor())
img, label = dataset[0]  # first test sample
img = img.view(-1).numpy()  # flatten to 784
q15 = np.clip(np.round(img * 32768), 0, 32767).astype(int)

with open('data/input_image.mem', 'w') as f:
    for val in q15:
        f.write(f"{val:016b}\n")

print("Saved input image to data/input_image.mem")
