import time
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import faiss

from scaling.imagenet.parameters.param_helper import get_name_of_global_linear_model
from scaling.models.evaluation import evaluate_model
from scaling.models.imagenet import load_linear_imagenet_model
from scaling.utils.data_loader import load_evaluation_set
from scaling.utils.data_loader_imagenet import load_clip_embeddings
from scaling.utils.directory import get_results_dir


def run_analysis_ttt_wrong_neighborhood(
        model_name,
        train_embeddings,
        train_labels,
        test_embeddings,
        test_labels,
        test_indices,
        num_neighbors,
        optimization,
        test_sample_index = 0,
):
    # Device configuration (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load the model
    global_model = load_linear_imagenet_model(device, model_name=model_name)
    global_model.to(device)

    # Iterate over optimization methods
    results = {}
    for opt_type, opt_params in optimization.items():
        # Get local model
        local_model = get_local_model(
            global_model=global_model,
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            test_embeddings=test_embeddings,
            test_sample_index=test_sample_index,
            num_neighbors=num_neighbors,
            opt_params=opt_params,
            device=device,
        )

        # Evaluate the local model on the entire test set
        accuracy, correct_list, ce_loss_list = evaluate_model(
            model=local_model,
            test_embeddings=test_embeddings[test_indices],
            test_labels=test_labels[test_indices],
        )

        # Initialize the results dictionary for this optimization type
        results[opt_type] = {
            'type': opt_type,
            'accuracy': accuracy,
            'correct_list': correct_list.tolist(),
            'ce_loss_list': ce_loss_list.tolist(),
        }

    # Evaluate the global model for reference
    accuracy_reference, correct_ref_list, ce_loss_list = evaluate_model(
        model=global_model,
        test_embeddings=test_embeddings[test_indices],
        test_labels=test_labels[test_indices],
    )

    # Add global model results to dictionary
    results['global'] = {
        'type': 'global',
        'accuracy': accuracy_reference,
        'correct_list': correct_ref_list.tolist(),
        'ce_loss_list': ce_loss_list.tolist(),
    }

    # Add additional parameters to results dictionary
    results['model_name'] = model_name
    results['num_samples'] = len(test_indices)
    results['num_neighbors'] = num_neighbors
    results['optimization'] = optimization

    return results


def get_local_model(
        global_model,
        train_embeddings,
        train_labels,
        test_embeddings,
        test_sample_index,
        num_neighbors,
        opt_params,
        device,
):
    # Build FAISS index
    faiss_index = faiss.IndexFlatL2(train_embeddings.shape[1])
    faiss_index.add(train_embeddings)

    # Fetch the k nearest neighbors
    test_sample_embedding = test_embeddings[test_sample_index].reshape(1, -1)
    _, indices = faiss_index.search(test_sample_embedding, num_neighbors)
    neighbor_indices = indices[0]
    neighbor_embeddings = train_embeddings[neighbor_indices]
    neighbor_labels = train_labels[neighbor_indices]

    # Sort neighbor by increasing distance to the test sample (for sequential method)
    distances = torch.norm(neighbor_embeddings - test_sample_embedding, dim=1)
    sorted_indices = torch.argsort(distances)
    neighbor_embeddings = neighbor_embeddings[sorted_indices]
    neighbor_labels = neighbor_labels[sorted_indices]

    # Local fine-tuning
    local_model = locally_fine_tune_torch_model(
        global_model=global_model,
        neighbor_embeddings=neighbor_embeddings,
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
    optimizer = torch.optim.Adam(local_model.parameters(), lr=optimization_params['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    # Move neighbor tensors to the appropriate device
    neighbor_tensors = neighbor_embeddings.to(device)
    neighbor_labels_tensors = neighbor_labels.long().to(device)

    # Run the fine-tuning loop for a few epochs
    if optimization_params['type'] == 'erm':
        for epoch in range(optimization_params['epochs']):
            # Forward pass
            outputs = local_model(neighbor_tensors)
            loss = criterion(outputs, neighbor_labels_tensors)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    elif optimization_params['type'] == 'sequential':
        for i in range(len(neighbor_tensors)):
            input_i = neighbor_tensors[i].unsqueeze(0)
            label_i = neighbor_labels_tensors[i].unsqueeze(0)

            # Forward pass
            output_i = local_model(input_i)
            loss = criterion(output_i, label_i)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    else:
        raise ValueError(f'Unknown optimization method: {optimization_params["type"]}')

    return local_model


if __name__ == '__main__':
    # Fix random seed for reproducibility
    np.random.seed(42)

    # Load the embeddings and labels
    print(f'Loading embeddings and labels...')
    train_embeddings, train_labels = load_clip_embeddings('training_data')
    test_embeddings, test_labels = load_clip_embeddings('test_data')
    val_embeddings, val_labels = load_clip_embeddings('validation_data')

    # Load the evaluation set
    evaluation_set = 'test'
    evaluation_embeddings, evaluation_labels, evaluation_indices = load_evaluation_set(
        test_embeddings,
        test_labels,
        val_embeddings,
        val_labels,
        evaluation_set=evaluation_set,
        num_indices_per_class='all'
    )

    # Parameters for local fine-tuning
    num_neighbors = 600

    # Optimization parameters
    optimization = {
        'erm': {
            'type': 'erm',
            'learning_rate': 2e-2,
            'epochs': 50,
        },
        'sequential': {
            'type': 'sequential',
            'learning_rate': 3 * 1e-3,
        },
    }

    # Model name
    model_name, cluster = get_name_of_global_linear_model()

    # Test sample to fine-tune on
    test_sample_index = 423

    print(f'Run TTT analysis...')
    start = time.time()
    results = run_analysis_ttt_wrong_neighborhood(
        model_name=model_name,
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        test_embeddings=evaluation_embeddings,
        test_labels=evaluation_labels,
        test_indices=evaluation_indices,
        num_neighbors=num_neighbors,
        optimization=optimization,
        test_sample_index=test_sample_index,
    )
    end = time.time()
    print(f'Finished in {end - start} seconds')

    print('\n' + '*' * 80)
    print(f'Accuracy of global model on {results["num_samples"]} samples: {(results["global"]["accuracy"] * 100):.2f}%')
    for opt_type, opt_params in optimization.items():
        print(f'Accuracy of locally fine-tuned linear classifier with "{opt_params["type"]}" optimization on {results["num_samples"]} samples: {(results[opt_type]["accuracy"] * 100):.2f}%')
    print(f'Test sample index: {test_sample_index}')
    print('*' * 80 + '\n')

    # Add parameters to results
    results['evaluation_set'] = evaluation_set
    results['test_sample_index'] = test_sample_index

    # Save results
    results_dir = get_results_dir(experiment_name='imagenet_ttt_wrong_neighborhood')
    results_file = f'{results_dir}/results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f)
    print(f'Results saved to {results_file}')