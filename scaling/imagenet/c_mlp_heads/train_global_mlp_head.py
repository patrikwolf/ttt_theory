import os
import time
import numpy as np
import wandb
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from scaling.models.torch_mlp_classifier import TorchMLPClassifier
from scaling.utils.data_loader import load_evaluation_set
from scaling.utils.data_loader_imagenet import load_clip_embeddings
from scaling.utils.directory import get_models_dir

def train_torch_mlp_classifier(
        train_tensors,
        train_labels,
        val_tensors,
        val_labels,
        learning_rate=1e-3,
        weight_decay=1e-5,
        batch_size=256,
        num_epochs=10,
        hidden_dim=1024,
        num_classes=1000,
        dropout_rate=0.5,
        use_wandb=True,
):
    print(f'Training MLP classifier with hidden dimension {hidden_dim}')

    # Device Configuration (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create Datasets and DataLoaders
    train_loader = DataLoader(dataset=TensorDataset(train_tensors, train_labels), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=TensorDataset(val_tensors, val_labels), batch_size=batch_size, shuffle=False)

    # Model, Loss, and Optimizer
    input_features = train_tensors.shape[1]
    num_classes = num_classes
    model = TorchMLPClassifier(input_features, hidden_dim, num_classes, dropout_rate).to(device)

    # CrossEntropyLoss is the standard for multi-class classification
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

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
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Accuracy: {accuracy:.2f}%')

        # Log to Weights & Biases
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'loss': loss.item(),
                'val_accuracy': accuracy,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'dropout_rate': dropout_rate,
                'weight_decay': weight_decay,
            })

    end_time = time.time()
    print(f'PyTorch training finished in {end_time - start_time:.2f} seconds.')

    # Attach classes_ attribute for compatibility with our helper functions
    model.classes_ = np.unique(train_labels)

    return model


if __name__ == '__main__':
    # Load the data
    train_embeddings, train_labels = load_clip_embeddings('training_data')
    test_embeddings, test_labels = load_clip_embeddings('test_data')
    val_embeddings, val_labels = load_clip_embeddings('validation_data')

    # Load the evaluation set
    evaluation_embeddings, evaluation_labels, _ = load_evaluation_set(
        test_embeddings,
        test_labels,
        val_embeddings,
        val_labels,
        evaluation_set='validation',
        num_indices_per_class='all'
    )

    # Hyperparameters
    hidden_dim = 750
    learning_rate = 3 * 1e-4
    weight_decay = 0.0
    batch_size = 256
    num_epochs = 50
    dropout_rate = 0.5

    # Initialize Weights & Biases
    wandb.init(project='mlp-classifier', config={
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'hidden_dim': hidden_dim,
        'dropout_rate': dropout_rate,
        'model': 'TorchMLPClassifier'
    })

    # Train the PyTorch MLP classifier
    torch_mlp_classifier = train_torch_mlp_classifier(
        train_tensors=train_embeddings,
        train_labels=train_labels,
        val_tensors=evaluation_embeddings,
        val_labels=evaluation_labels,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        num_epochs=num_epochs,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
        use_wandb=True,
    )

    # Save the PyTorch model
    model_dir = get_models_dir()
    model_name = (f'torch_mlp_classifier_lr{learning_rate:.4f}_bs{batch_size}_hd{hidden_dim}_wd{weight_decay:.4f}_'
                  f'dr{dropout_rate:.2f}_ne{num_epochs}.pth')
    torch.save(torch_mlp_classifier.state_dict(), f'{model_dir}/{model_name}')

    print('*' * 80)
    print(f'PyTorch model for MLP classifier with hidden dimension {hidden_dim} saved as {model_name}')
    print('*' * 80)

    print(f'Model trained with file "{os.path.basename(__file__)}"')