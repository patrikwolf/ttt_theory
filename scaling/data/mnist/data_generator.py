import os
import torch
import torchvision
import torchvision.transforms as transforms

from pathlib import Path
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def save_mnist_tensors():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '_data')

    # Transformation
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))])

    # Load the MNIST dataset
    download = True
    train_val_set = torchvision.datasets.MNIST(root=data_dir, train=True, download=download, transform=transform)
    test_set = torchvision.datasets.MNIST(root=data_dir, train=False, download=download, transform=transform)

    # Train / validation split
    train_val_indices = list(range(len(train_val_set)))
    train_val_targets = [train_val_set[i][1] for i in train_val_indices]
    train_indices, val_indices = train_test_split(
        train_val_indices,
        train_size=50_000,
        test_size=10_000,
        random_state=0,
        shuffle=True,
        stratify=train_val_targets,
    )
    train_subset = Subset(train_val_set, train_indices)
    val_subset = Subset(train_val_set, val_indices)

    print(f'Size of train subset: {len(train_subset)}')
    print(f'Size of validation subset: {len(val_subset)}')
    print(f'Size of test set: {len(test_set)}')

    train_images = torch.stack([train_subset[i][0] for i in range(len(train_subset))])
    train_labels = torch.tensor([train_subset[i][1] for i in range(len(train_subset))], dtype=torch.long)

    val_images = torch.stack([val_subset[i][0] for i in range(len(val_subset))])
    val_labels = torch.tensor([val_subset[i][1] for i in range(len(val_subset))], dtype=torch.long)

    test_images = torch.stack([test_set[i][0] for i in range(len(test_set))])
    test_labels = torch.tensor([test_set[i][1] for i in range(len(test_set))], dtype=torch.long)

    # Save the tensors
    save_path = Path(f'{base_dir}/images/training_data.pt')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'images': train_images,
        'labels': train_labels
    }, save_path)

    save_path = Path(f'{base_dir}/images/validation_data.pt')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'images': val_images,
        'labels': val_labels
    }, save_path)

    save_path = Path(f'{base_dir}/images/test_data.pt')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'images': test_images,
        'labels': test_labels
    }, save_path)


def load_mnist_images(file_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    torch_data = torch.load(f'{base_dir}/images/{file_name}.pt')
    return torch_data['images'], torch_data['labels']


if __name__ == '__main__':
    # Save the MNIST tensors to a file
    save_mnist_tensors()

    # Load the images and labels
    train_images, train_labels = load_mnist_images('training_data')
    print(f'Shape of training images: {train_images.shape}, labels: {train_labels.shape}')

