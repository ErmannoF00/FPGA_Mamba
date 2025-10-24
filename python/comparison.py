import torch
from torchvision import datasets, transforms
import numpy as np

# Load true label
dataset = datasets.MNIST('.', download=True, train=False, transform=transforms.ToTensor())
img, true_label = dataset[0]

# Load Verilog output
with open("output.txt") as f:
    out_vals = [int(line.strip()) for line in f.readlines()]

predicted = np.argmax(out_vals)

print("True label:     ", true_label)
print("Predicted class:", predicted)

if predicted == true_label:
    print("✅ MATCH")
else:
    print("❌ MISMATCH")
