import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def sample_of_classes():
    # image --> tensor
    # normalize pixel
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # MNIST dataset 
    dataset = datasets.MNIST('../Thedata', train=True, download=True, transform=transform)

    samples = {}

    for image, label in dataset:
        if label not in samples:
            samples[label] = image
        if len(samples) == 10:  
            break

    fig, axes = plt.subplots(1, 10, figsize=(15, 1.5))
    for i in range(10):
        axes[i].imshow(samples[i].squeeze(), cmap='gray')
        axes[i].set_title(f'{i}')
        axes[i].axis('off')

    plt.show()


sample_of_classes()