import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def sample_of_classes():
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))  
    ])

    # load MNIST dataset
    Dataset = datasets.MNIST(root='../Thedata', train=True, download=True, transform=transform)

    # 10 classes --> 0 to 9
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


    plt.figure(figsize=(10, 5))
    for k in range(len(classes)):
        plt.subplot(2, 5, k + 1)
        sample_image = Dataset.data[Dataset.targets == k][0]
        plt.imshow(sample_image, cmap='gray')
        plt.title(f'Class {classes[k]}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def split_train_val(dataset, val_split=0.2):
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

if __name__ == "__main__":

    sample_of_classes()
    