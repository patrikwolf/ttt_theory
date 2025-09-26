import os
import torch

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def load_mnist_embeddings(file_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    torch_data = torch.load(f'{base_dir}/../data/MNIST/embeddings/{file_name}.pt')

    # Extract metadata if available
    meta = {
        'model_name': torch_data.get('model_name', None),
        'timestamp': torch_data.get('timestamp', None),
    }

    return torch_data['images'], torch_data['embeddings'], torch_data['labels'], meta


if __name__ == '__main__':
    # Load the data
    train_images, train_embeddings, train_labels, _ = load_mnist_embeddings('training_data')
    print(f'Loaded {len(train_images)} images and {len(train_labels)} labels.')

