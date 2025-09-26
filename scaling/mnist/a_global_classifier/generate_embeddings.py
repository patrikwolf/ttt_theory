import os
import json
import numpy as np
import torch

from datetime import datetime
from pathlib import Path
from scaling.data.mnist.data_generator import load_mnist_images
from scaling.mnist.parameters.param_helper import get_name_of_global_mnist_model
from scaling.models.mnist import load_global_mnist_model
from scaling.utils.data_loader_mnist import load_mnist_embeddings


def generate_last_layer_embeddings(
        model_name,
        train_images,
        train_labels,
        test_images,
        test_labels,
        val_images,
        val_labels,
        size_scale=0.5,
        cluster=False,
):
    # Device configuration (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load the model
    global_model = load_global_mnist_model(
        device=device,
        model_name=model_name,
        size_scale=size_scale,
        cluster=cluster,
        only_head=False,
    )

    # Generate last-hidden-layer embeddings
    print('Generating last layer embeddings for train set...')
    train_embeddings = global_model(train_images, extract_last_layer=True)

    print('Generating last layer embeddings for test set...')
    test_embeddings = global_model(test_images, extract_last_layer=True)

    print('Generating last layer embeddings for validation set...')
    val_embeddings = global_model(val_images, extract_last_layer=True)

    # Detach embeddings from the computation graph
    train_embeddings = train_embeddings.detach()
    test_embeddings = test_embeddings.detach()
    val_embeddings = val_embeddings.detach()

    # Save embeddings
    now = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    embd_dir = os.path.join(base_dir, '../../data/MNIST/embeddings')
    save_path = Path(f'{embd_dir}/training_data.pt')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Saving embeddings to {save_path}')

    torch.save({
        'images': train_images,
        'embeddings': train_embeddings,
        'labels': train_labels,
        'model_name': model_name,
        'timestamp': now,
    }, save_path)

    save_path = Path(f'{embd_dir}/test_data.pt')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Saving embeddings to {save_path}')
    torch.save({
        'images': test_images,
        'embeddings': test_embeddings,
        'labels': test_labels,
        'model_name': model_name,
        'timestamp': now,
    }, save_path)

    save_path = Path(f'{embd_dir}/validation_data.pt')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Saving embeddings to {save_path}')
    torch.save({
        'images': val_images,
        'embeddings': val_embeddings,
        'labels': val_labels,
        'model_name': model_name,
        'timestamp': now,
    }, save_path)

    print('Done!')


if __name__ == '__main__':
    # Fix random seed for reproducibility
    np.random.seed(42)

    # Load the images and labels
    print(f'Loading images and labels...')
    train_images, train_labels = load_mnist_images('training_data')
    test_images, test_labels = load_mnist_images('test_data')
    val_images, val_labels = load_mnist_images('validation_data')

    # Run the analysis
    model_name, size_scale, cluster = get_name_of_global_mnist_model()

    # Generate last layer embeddings
    generate_last_layer_embeddings(
        model_name=model_name,
        size_scale=size_scale,
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        val_images=val_images,
        val_labels=val_labels,
        cluster=cluster,
    )

    # Load the images, embeddings and labels
    print(f'Loading images, embeddings and labels...')
    train_images, train_embeddings, train_labels, meta_train = load_mnist_embeddings('training_data')
    test_images, test_embeddings, test_labels, meta_test = load_mnist_embeddings('test_data')
    val_images, val_embeddings, val_labels, meta_val = load_mnist_embeddings('validation_data')

    print(json.dumps(meta_test, indent=4))