import time
import json
import numpy as np
import wandb
import torch
import torch.nn as nn
import argparse
import itertools

from torch.utils.data import DataLoader, TensorDataset
from scaling.mnist.b_linear_head.eval_mnist_linear_head import run_analysis_mnist_linear_head
from scaling.models.evaluation import evaluate_model
from scaling.models.torch_linear_classifier import TorchLinearClassifier
from scaling.utils.data_loader import load_evaluation_set
from scaling.utils.data_loader_mnist import load_mnist_embeddings
from scaling.utils.directory import get_models_dir, get_results_dir


def train_mnist_linear_head(
        train_embeddings,
        train_labels,
        val_embeddings,
        val_labels,
        learning_rate,
        batch_size,
        num_epochs,
        num_classes = 10,
        use_wandb=True,
):
    # Device configuration (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create Datasets and DataLoaders
    train_loader = DataLoader(dataset=TensorDataset(train_embeddings, train_labels), batch_size=batch_size, shuffle=True)

    # Model, Loss, and Optimizer
    input_features = train_embeddings.shape[1]
    model = TorchLinearClassifier(input_features, num_classes).to(device)

    # CrossEntropyLoss is the standard for multi-class classification
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
        val_embeddings = val_embeddings.to(device)
        val_labels = val_labels.to(device)
        accuracy, _, _ = evaluate_model(model, val_embeddings, val_labels)
        print(f'Epoch [{epoch+1}/{num_epochs}], training loss: {loss.item():.2f}, validation accuracy: {100 * accuracy:.2f}%')

        # Log to Weights & Biases
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'loss': loss.item(),
                'val_accuracy': accuracy
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
    _, train_embeddings, train_labels, _ = load_mnist_embeddings('training_data')
    _, test_embeddings, test_labels, _ = load_mnist_embeddings('test_data')
    _, val_embeddings, val_labels, _ = load_mnist_embeddings('validation_data')

    # Load the evaluation set (this evaluation set is only used for W&B)
    evaluation_embeddings, evaluation_labels, evaluation_indices = load_evaluation_set(
        test_embeddings,
        test_labels,
        val_embeddings,
        val_labels,
        evaluation_set='validation',
        num_indices_per_class='all'
    )

    # Hyperparameters
    hyperparameters = {}
    count = 0
    learning_rates = [5e-3, 1e-3, 8e-4, 5e-4, 4e-4]
    batch_sizes = [30, 50, 75, 100, 150, 200, 250, 300, 400, 500]
    num_epochs = [2, 50, 100]

    for lr, bs, ne in itertools.product(learning_rates, batch_sizes, num_epochs):
        hyperparameters[f'shard_{count}'] = {
            'learning_rate': lr,
            'batch_size': bs,
            'num_epochs': ne,
        }
        count += 1

    # Select hyperparameters for the current shard
    shard = f'shard_{args.shard_id}'

    # Initialize Weights & Biases
    wandb.init(project='mnist-linear-head', config={
        'learning_rate': hyperparameters[shard]['learning_rate'],
        'batch_size': hyperparameters[shard]['batch_size'],
        'num_epochs': hyperparameters[shard]['num_epochs'],
        'model': 'TorchLinearClassifier'
    })

    # Train the PyTorch linear classifier
    start = time.time()
    torch_classifier = train_mnist_linear_head(
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        val_embeddings=evaluation_embeddings,
        val_labels=evaluation_labels,
        learning_rate=hyperparameters[shard]['learning_rate'],
        batch_size=hyperparameters[shard]['batch_size'],
        num_epochs=hyperparameters[shard]['num_epochs'],
        use_wandb=False,
    )

    # Time
    end = time.time()
    print(f'\nTraining time: {end - start:.2f} seconds.\n')

    # Finish Weights & Biases run
    wandb.finish()

    # Save the PyTorch model
    model_dir = get_models_dir()
    model_name = (f'mnist_linear_classifier_'
                  f'lr{hyperparameters[shard]["learning_rate"]}'
                  f'bs{hyperparameters[shard]["batch_size"]}_'
                  f'ne{hyperparameters[shard]["num_epochs"]}')
    torch.save(torch_classifier.state_dict(), f'{model_dir}/{model_name}.pth')

    print('*' * 80)
    print(f'PyTorch linear classifier model saved as {model_name}')
    print('*' * 80)

    # Evaluate the model on the test set
    results_test = run_analysis_mnist_linear_head(
        model_name=model_name,
        test_embeddings=test_embeddings,
        test_labels=test_labels,
        save_results=False,
    )

    # Evaluate the model on the validation set
    results_val = run_analysis_mnist_linear_head(
        model_name=model_name,
        test_embeddings=val_embeddings,
        test_labels=val_labels,
        save_results=False,
    )

    print('*' * 80)
    print(f'Global linear head achieves a test accuracy of: {(results_test["accuracy"] * 100):.2f}%')
    print(f'Global linear head achieves a validation accuracy of: {(results_val["accuracy"] * 100):.2f}%')
    print('*' * 80)

    # Add parameters to the results
    results_test['accuracy_test'] = results_test['accuracy']
    results_test['accuracy_val'] = results_val['accuracy']
    results_test['learning_rate'] = hyperparameters[shard]['learning_rate']
    results_test['batch_size'] = hyperparameters[shard]['batch_size']
    results_test['num_epochs'] = hyperparameters[shard]['num_epochs']

    # Save results
    results_dir = get_results_dir(experiment_name='mnist_global_linear', timestamp=args.datetime)
    results_file = f'{results_dir}/shard_{args.shard_id}.json'
    with open(results_file, 'w') as f:
        json.dump(results_test, f)
    print(f'Results saved to {results_file}')