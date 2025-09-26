import time
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import faiss

from scaling.mnist.parameters.param_helper import get_optimal_mnist_cnn_parameters
from scaling.mnist.parameters.param_helper import get_optimal_lft_cnn_parameters
from scaling.models.evaluation import evaluate_model
from scaling.models.mnist import load_global_mnist_model
from scaling.utils.data_loader import load_evaluation_set
from scaling.utils.data_loader_mnist import load_mnist_embeddings
from scaling.utils.directory import get_results_dir


def run_analysis_lft_wrong_neighborhood(
        model_name,
        size_scale,
        hidden_dim,
        train_images,
        train_labels,
        test_images,
        test_labels,
        test_indices,
        optimization,
        test_sample_index,
        model_on_cluster=True,
):
    # Device configuration (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load the model
    scaled_cnn = load_global_mnist_model(
        device=device,
        model_name=model_name,
        size_scale=size_scale,
        cluster=model_on_cluster,
        only_head=False,
    )
    scaled_cnn.to(device)

    # Number of parameters
    total_params = sum(p.numel() for p in scaled_cnn.parameters())

    # Map the embeddings to the feature space
    train_features = scaled_cnn(train_images.to(device), extract_last_layer=True).detach()
    test_features = scaled_cnn(test_images.to(device), extract_last_layer=True).detach()

    # Get the last linear layer
    last_linear_layer = scaled_cnn.classifier

    # Evaluate the global model for reference
    accuracy_reference, correct_ref_list, ce_loss_list = evaluate_model(
        model=last_linear_layer,
        test_embeddings=test_features[test_indices],
        test_labels=test_labels[test_indices],
    )

    # Add global model results to dictionary
    results = {}
    results['global'] = {
        'type': 'global',
        'accuracy': accuracy_reference,
        'correct_list': correct_ref_list.tolist(),
        'ce_loss_list': ce_loss_list.tolist(),
    }

    # Get local model
    local_model = get_local_model(
        global_model=last_linear_layer,
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_sample_index=test_sample_index,
        opt_params=optimization,
        device=device,
    )

    # Evaluate the local model on the entire test set
    accuracy, correct_list, ce_loss_list = evaluate_model(
        model=local_model,
        test_embeddings=test_features[test_indices],
        test_labels=test_labels[test_indices],
    )

    # Initialize the results dictionary for this optimization type
    results['lft'] = {
        'accuracy': accuracy,
        'correct_list': correct_list.tolist(),
        'ce_loss_list': ce_loss_list.tolist(),
    }

    # Add additional parameters to results dictionary
    results['model_name'] = model_name
    results['size_scale'] = size_scale
    results['hidden_dim'] = hidden_dim
    results['total_params'] = total_params
    results['num_samples'] = len(test_indices)
    results['optimization'] = optimization

    return results


def get_local_model(
        global_model,
        train_features,
        train_labels,
        test_features,
        test_sample_index,
        opt_params,
        device,
):
    # Build FAISS index
    faiss_index = faiss.IndexFlatL2(train_features.shape[1])
    faiss_index.add(train_features.detach().cpu().numpy())

    # Get the feature of the test sample to fine-tune on
    test_feature = test_features[test_sample_index].reshape(1, -1)

    # Fetch the k nearest neighbors
    num_neighbors = opt_params['num_neighbors']
    _, indices = faiss_index.search(test_feature, num_neighbors)
    neighbor_indices = indices[0]
    neighbor_features = train_features[neighbor_indices]
    neighbor_labels = train_labels[neighbor_indices]

    # Local fine-tuning
    local_model = locally_fine_tune_torch_model(
        global_model=global_model,
        neighbor_embeddings=neighbor_features,
        neighbor_labels=neighbor_labels,
        device=device,
        optimization_params=opt_params,
    )

    return local_model


def locally_fine_tune_torch_model(
        global_model,
        neighbor_embeddings,
        neighbor_labels,
        device,
        optimization_params,
):
    # Create a deep copy of the global model to fine-tune.
    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()

    # Set up a new optimizer and loss function for the local model
    optimizer = torch.optim.Adam(local_model.parameters(), lr=optimization_params['finetune_lr'])
    criterion = nn.CrossEntropyLoss()

    # Move neighbor tensors to the appropriate device
    neighbor_tensors = neighbor_embeddings.to(device)
    neighbor_labels_tensors = neighbor_labels.long().to(device)

    # Run the fine-tuning loop for a few epochs
    for epoch in range(optimization_params['finetune_epochs']):
        # Forward pass
        outputs = local_model(neighbor_tensors)
        loss = criterion(outputs, neighbor_labels_tensors)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return local_model


if __name__ == '__main__':
    # Fix random seed for reproducibility
    np.random.seed(42)

    # Load the embeddings and labels
    print(f'Loading embeddings and labels...')
    train_images, _, train_labels, _ = load_mnist_embeddings('training_data')
    test_images, _, test_labels, _ = load_mnist_embeddings('test_data')
    val_images, _, val_labels, _ = load_mnist_embeddings('validation_data')

    # Load the evaluation set
    evaluation_set = 'test'
    evaluation_images, evaluation_labels, evaluation_indices = load_evaluation_set(
        test_images,
        test_labels,
        val_images,
        val_labels,
        evaluation_set=evaluation_set,
        num_indices_per_class='all'
    )

    # Parameters for the CNN head
    seed = 1
    hidden_dim = 25
    size_scale = 0.35
    model_on_cluster = False
    models = get_optimal_mnist_cnn_parameters()
    model_name = (f'mnist_scaled_model_'
                  f'seed{seed}_'
                  f'hd{hidden_dim}_'
                  f'ss{size_scale}_'
                  f'ot-{models[str(hidden_dim)]["optimization_type"]}_'
                  f'lr{models[str(hidden_dim)]["learning_rate"]}'
                  f'bs{models[str(hidden_dim)]["batch_size"]}_'
                  f'ne{models[str(hidden_dim)]["num_epochs"]}')

    # Parameters for local fine-tuning
    lft_params = get_optimal_lft_cnn_parameters()
    optimization = {
        'num_neighbors': lft_params[str(hidden_dim)]['num_neighbors'],
        'finetune_epochs': lft_params[str(hidden_dim)]['finetune_epochs'],
        'finetune_lr': lft_params[str(hidden_dim)]['finetune_lr'],
    }

    # Test sample to fine-tune on
    test_sample_index = 762

    print(f'Run LFT analysis...')
    start = time.time()
    results = run_analysis_lft_wrong_neighborhood(
        model_name=model_name,
        size_scale=size_scale,
        hidden_dim=hidden_dim,
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        test_indices=evaluation_indices,
        optimization=optimization,
        test_sample_index=test_sample_index,
        model_on_cluster=model_on_cluster,
    )
    end = time.time()
    print(f'Finished in {end - start} seconds')

    print('\n' + '*' * 80)
    print(f'Accuracy of global model on {results["num_samples"]} samples: {(results["global"]["accuracy"] * 100):.2f}%')
    print(f'Accuracy of locally fine-tuned linear classifier on {results["num_samples"]} samples: {(results["lft"]["accuracy"] * 100):.2f}%')
    print(f'Test sample index: {test_sample_index}')
    print('*' * 80 + '\n')

    # Add parameters to results
    results['evaluation_set'] = evaluation_set
    results['test_sample_index'] = test_sample_index

    # Save results
    results_dir = get_results_dir(experiment_name='mnist_lft_wrong_neighborhood')
    results_file = f'{results_dir}/results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f)
    print(f'Results saved to {results_file}')