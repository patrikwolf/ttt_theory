import time
import json
import numpy as np
import wandb
import torch
import torch.nn as nn
import argparse
import itertools

from torch.utils.data import DataLoader, TensorDataset
from scaling.mnist.a_global_classifier.eval_reference import evaluate_mnist_model, run_analysis_global_minst
from scaling.models.mnist import load_raw_mnist_model
from scaling.utils.data_loader_mnist import load_mnist_embeddings
from scaling.utils.directory import get_models_dir, get_results_dir


def train_global_mnist_classifier(
        model_type,
        train_images,
        train_labels,
        val_images,
        val_labels,
        learning_rate,
        momentum,
        weight_decay,
        batch_size,
        num_epochs,
        optimization_type='sgd',
        num_classes=10,
        use_wandb=True,
):
    # Device configuration (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create Datasets and DataLoaders
    train_loader = DataLoader(dataset=TensorDataset(train_images, train_labels), batch_size=batch_size, shuffle=True)

    # Model, Loss, and Optimizer
    model = load_raw_mnist_model(model_type, num_classes)
    model.to(device)

    # CrossEntropyLoss is the standard for multi-class classification
    criterion = nn.CrossEntropyLoss()
    if optimization_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimization_type == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError('Unknown optimization type')

    # Training Loop
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        for i, (features, labels) in enumerate(train_loader):
            features = features.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation after each epoch
        val_images = val_images.to(device)
        val_labels = val_labels.to(device)
        results = evaluate_mnist_model(model, val_images, val_labels)
        print(f'Epoch [{epoch+1}/{num_epochs}], training loss: {loss.item():.2f}, validation accuracy: '
              f'{100 * results["accuracy"]:.2f}%')

        # Log to Weights & Biases
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'loss': loss.item(),
                'val_accuracy': results['accuracy'],
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'momentum': momentum,
                'weight_decay': weight_decay,
                'optimization_type': optimization_type,
            })

    end_time = time.time()
    print(f'PyTorch training finished in {end_time - start_time:.2f} seconds.')

    # Attach classes_ attribute for compatibility with our helper functions
    model.classes_ = np.unique(train_labels)

    return model


if __name__ == '__main__':
    # Fix random seed for reproducibility
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_id", type=int, required=True)
    parser.add_argument("--datetime", type=str, required=True)
    args = parser.parse_args()

    # Load the data (test set = artificial)
    print(f'Loading embeddings and labels...')
    train_images, _, train_labels, _ = load_mnist_embeddings('training_data')
    test_images, _, test_labels, _ = load_mnist_embeddings('test_data')
    val_images, _, val_labels, _ = load_mnist_embeddings('validation_data')

    # Hyperparameters
    hyperparameters = {}
    count = 0
    optimization_type = ['adam', 'sgd']
    learning_rates = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    batch_sizes = [50, 100, 200, 300, 400]
    num_epochs = [2]
    momentum = 0.9              # Not used for Adam
    weight_decay = 0.0001       # Not used for Adam

    for ot, lr, bs, ne in itertools.product(optimization_type, learning_rates, batch_sizes, num_epochs):
        hyperparameters[f'shard_{count}'] = {
            'optimization_type': ot,
            'learning_rate': lr,
            'batch_size': bs,
            'num_epochs': ne,
            'momentum': momentum,            # Not used for Adam
            'weight_decay': weight_decay,    # Not used for Adam
        }
        count += 1

    # Select hyperparameters for the current shard
    shard = f'shard_{args.shard_id}'

    # Model type
    size_scale = 0.5
    model_type = f'FlexibleLeNet_{size_scale:.2f}'

    # Initialize Weights & Biases
    use_wandb = True
    if use_wandb:
        wandb.init(project='mnist-global-classifier', config={
            'optimization_type': hyperparameters[shard]['optimization_type'],
            'learning_rate': hyperparameters[shard]['learning_rate'],
            'batch_size': hyperparameters[shard]['batch_size'],
            'num_epochs': hyperparameters[shard]['num_epochs'],
            'momentum': hyperparameters[shard]['momentum'],
            'weight_decay': hyperparameters[shard]['weight_decay'],
            'model': model_type,
        })

    # Train the PyTorch linear classifier
    torch_classifier = train_global_mnist_classifier(
        model_type=model_type,
        train_images=train_images,
        train_labels=train_labels,
        val_images=val_images,
        val_labels=val_labels,
        learning_rate=hyperparameters[shard]['learning_rate'],
        momentum=hyperparameters[shard]['momentum'],
        weight_decay=hyperparameters[shard]['weight_decay'],
        batch_size=hyperparameters[shard]['batch_size'],
        num_epochs=hyperparameters[shard]['num_epochs'],
        optimization_type=hyperparameters[shard]['optimization_type'],
        use_wandb=use_wandb,
    )

    if use_wandb:
        # Finish the Weights & Biases run
        wandb.finish()

    # Save the PyTorch model
    model_dir = get_models_dir()
    model_name = (f'torch_global_classifier_mnist_'
                  f'ot-{hyperparameters[shard]["optimization_type"]}_'
                  f'lr{hyperparameters[shard]["learning_rate"]}'
                  f'bs{hyperparameters[shard]["batch_size"]}_'
                  f'ne{hyperparameters[shard]["num_epochs"]}')
    if hyperparameters[shard]["optimization_type"] == "sgd":
        sgd_suffix = (
            f'_wd{hyperparameters[shard]["weight_decay"]:.4f}'
            f'_mo{hyperparameters[shard]["momentum"]:.2f}'
        )
        model_name = model_name + sgd_suffix
    torch.save(torch_classifier.state_dict(), f'{model_dir}/{model_name}.pth')

    print('*' * 80)
    print(f'PyTorch linear classifier model saved as {model_name}')
    print('*' * 80)

    # Evaluate the model
    results = run_analysis_global_minst(
        model_name=model_name,
        size_scale=size_scale,
        test_images=test_images,
        test_labels=test_labels,
        val_images=val_images,
        val_labels=val_labels,
        cluster=False,
        save_results=False,
    )

    print('*' * 80)
    print(f'Global linear head achieves a test accuracy of: {(results["accuracy_test"] * 100):.2f}%')
    print(f'Global linear head achieves a validation accuracy of: {(results["accuracy_val"] * 100):.2f}%')
    print('*' * 80)

    # Add parameters to the results
    results['learning_rate'] = hyperparameters[shard]['learning_rate']
    results['batch_size'] = hyperparameters[shard]['batch_size']
    results['num_epochs'] = hyperparameters[shard]['num_epochs']
    results['optimization_type'] = hyperparameters[shard]['optimization_type']
    results['momentum'] = hyperparameters[shard]['momentum']
    results['weight_decay'] = hyperparameters[shard]['weight_decay']

    # Save results
    results_dir = get_results_dir(experiment_name='mnist_global_classifier', timestamp=args.datetime)
    results_file = f'{results_dir}/shard_{args.shard_id}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f)
    print(f'Results saved to {results_file}')