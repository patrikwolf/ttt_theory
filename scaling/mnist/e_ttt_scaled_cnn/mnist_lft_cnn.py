import os
import time
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import faiss
import wandb

from scaling.log_book.read_and_write import log_to_csv, print_tabulated
from scaling.mnist.parameters.param_helper import get_optimal_mnist_cnn_parameters
from scaling.models.evaluation import evaluate_model
from scaling.models.mnist import load_global_mnist_model
from scaling.utils.data_loader import load_evaluation_set
from scaling.utils.data_loader_mnist import load_mnist_embeddings
from scaling.utils.directory import get_results_dir


def run_analysis_lft_cnn_head(
        model_name,
        size_scale,
        hidden_dim,
        train_images,
        train_labels,
        test_images,
        test_labels,
        test_indices,
        num_neighbors,
        optimization,
        model_on_cluster=True,
        save_results=True,
        use_wandb=False,
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

    # Build FAISS index
    faiss_index = faiss.IndexFlatL2(train_features.shape[1])
    faiss_index.add(train_features.detach().cpu().numpy())

    print('Evaluating locally fine-tuned model...')
    results_dict = evaluate_model_with_lft(
        torch_classifier=last_linear_layer,
        device=device,
        faiss_index=faiss_index,
        train_embeddings=train_features,
        train_labels=train_labels,
        test_embeddings=test_features,
        test_labels=test_labels,
        test_indices=test_indices,
        num_neighbors=num_neighbors,
        optimization=optimization,
        use_wandb=use_wandb,
    )

    # Add parameters to results dictionary
    results_dict['model_name'] = model_name
    results_dict['total_params'] = total_params
    results_dict['hidden_dim'] = hidden_dim
    results_dict['num_samples'] = len(test_indices)
    results_dict['num_neighbors'] = num_neighbors
    results_dict['finetune_epochs'] = optimization['finetune_epochs']
    results_dict['finetune_lr'] = optimization['finetune_lr']

    # Save the results to a JSON file
    if save_results:
        results_dir = get_results_dir(experiment_name='mnist_lft_cnn')
        with open(f'{results_dir}/results.json', 'w') as f:
            json.dump(results_dict, f)
        print(f'Results saved to {results_dir}/results.json')
    else:
        print(f'---> Result saving within evaluation function is disabled.')

    return results_dict


def evaluate_model_with_lft(
        torch_classifier,
        device,
        faiss_index,
        train_embeddings,
        train_labels,
        test_embeddings,
        test_labels,
        test_indices,
        num_neighbors,
        optimization,
        use_wandb,
):
    # Initialize counters for correct predictions
    num_samples = len(test_indices)
    stats = {
        'correct_list': np.zeros(num_samples, dtype=int),
        'ce_loss_list': np.zeros(num_samples),
    }
    results_dict = {
        'global': copy.deepcopy(stats),
        'lft': copy.deepcopy(stats),
    }

    start = time.time()
    for idx, test_idx in enumerate(test_indices):
        if (idx + 1) % 100 == 0:
            print(f'Evaluating sample {idx + 1}/{num_samples} ({100 * (idx + 1) / num_samples:.2f}%) after '
                  f'{((time.time() - start) / 60):.2f} minutes...')

        # Get the test sample embedding and its true label
        test_sample_embedding = test_embeddings[test_idx].reshape(1, -1)
        true_label = test_labels[test_idx]
        test_label = true_label.to(device).unsqueeze(0)

        # Make prediction with the global model
        test_tensor = test_sample_embedding.to(device)
        predicted_label, _, ce_loss = evaluate_model(
            model=torch_classifier,
            test_embeddings=test_tensor,
            test_labels=test_label,
        )

        results_dict['global']['correct_list'][idx] = int(predicted_label == true_label)
        results_dict['global']['ce_loss_list'][idx] = ce_loss.item()

        # Use Weights & Biases only for the first sample
        use_wandb = (use_wandb and idx == 0)

        # Get prediction from the local fine-tuning method
        results = locally_fine_tune_and_predict_torch(
            test_sample_embedding,
            true_label,
            torch_classifier,
            train_embeddings,
            train_labels,
            faiss_index,
            device=device,
            num_neighbors=num_neighbors,
            optimization=optimization,
            use_wandb=use_wandb,
        )

        # Update the results dictionary
        results_dict['lft']['correct_list'][idx] = int(results['predicted_label'] == true_label)
        results_dict['lft']['ce_loss_list'][idx] = results['ce_loss']

        if (idx + 1) % 100 == 0:
            print(f'Current accuracy of global MLP head: {(100 * np.sum(results_dict["global"]["correct_list"]) / (idx + 1)):.2f}')
            print(f'Current accuracy of locally fine-tuned MLP head with ERM optimization: {(100 * np.sum(results_dict["lft"]["correct_list"]) / (idx + 1)):.2f}')
            print('*' * 100)

    # Calculate the accuracy for both global and local fine-tuning
    results_dict['global']['accuracy'] = float(np.sum(results_dict['global']['correct_list']) / num_samples)
    results_dict['global']['correct_list'] = results_dict['global']['correct_list'].tolist()
    results_dict['global']['ce_loss_list'] = results_dict['global']['ce_loss_list'].tolist()

    results_dict['lft']['accuracy'] = float(np.sum(results_dict['lft']['correct_list']) / num_samples)
    results_dict['lft']['correct_list'] = results_dict['lft']['correct_list'].tolist()
    results_dict['lft']['ce_loss_list'] = results_dict['lft']['ce_loss_list'].tolist()

    return results_dict


def locally_fine_tune_and_predict_torch(
        test_sample,
        test_label,
        torch_classifier,
        train_embeddings,
        train_labels,
        faiss_index,
        device,
        num_neighbors,
        optimization,
        use_wandb,
):
    # Fetch the k nearest neighbors
    _, indices = faiss_index.search(test_sample.detach().cpu().numpy(), num_neighbors)
    neighbor_indices = indices[0]
    neighbor_embeddings = train_embeddings[neighbor_indices]
    neighbor_labels = train_labels[neighbor_indices].to(device)

    # Sort neighbor by increasing distance to the test sample
    distances = torch.norm(neighbor_embeddings - test_sample, dim=1)
    sorted_indices = torch.argsort(distances)
    neighbor_embeddings = neighbor_embeddings[sorted_indices]
    neighbor_labels = neighbor_labels[sorted_indices]

    # Move the data to the appropriate device
    test_tensor = test_sample.to(device)
    test_label = test_label.to(device).unsqueeze(0)

    # Local fine-tuning
    local_model = locally_fine_tune_torch_model(
        global_model=torch_classifier,
        neighbor_embeddings=neighbor_embeddings,
        neighbor_labels=neighbor_labels,
        device=device,
        optimization_params=optimization,
        use_wandb=use_wandb,
    )

    # Make the final prediction using the fine-tuned local model
    predicted_label, _, ce_loss = evaluate_model(
        model=local_model,
        test_embeddings=test_tensor,
        test_labels=test_label,
    )

    # Save results in a dictionary
    results = {
        'predicted_label': predicted_label,
        'ce_loss': ce_loss.item(),
    }

    return results


def locally_fine_tune_torch_model(
        global_model,
        neighbor_embeddings,
        neighbor_labels,
        device,
        optimization_params,
        use_wandb,
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

        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'loss': loss.item(),
            })

    return local_model


if __name__ == '__main__':
    print('*' * 80)
    print('Running LFT on the last linear layer of the scaled CNN models.')
    print('*' * 80)

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

    # Parameters for local fine-tuning
    optimization = {
        'num_neighbors': 10,
        'finetune_epochs': 50,
        'finetune_lr': 1 * 1e-2,
    }

    # Parameters for the CNN
    hidden_dim = 25
    models = get_optimal_mnist_cnn_parameters()
    size_scale = models[str(hidden_dim)]['size_scale']
    optimization_type = models[str(hidden_dim)]['optimization_type']
    learning_rate = models[str(hidden_dim)]['learning_rate']
    batch_size = models[str(hidden_dim)]['batch_size']
    num_epochs = models[str(hidden_dim)]['num_epochs']

    # Model name
    model_name = (f'mnist_scaled_model_'
                  f'hd{hidden_dim}_'
                  f'ss{size_scale}_'
                  f'ot-{optimization_type}_'
                  f'lr{learning_rate}'
                  f'bs{batch_size}_'
                  f'ne{num_epochs}')
    model_on_cluster = False

    # Weights & Biases
    use_wandb = True
    if use_wandb:
        # Initialize W&B run
        wandb.init(project=f'mnist-lft-cnn-{size_scale}', config={
            'num_neighbors':optimization['num_neighbors'],
            'finetune_epochs': optimization['finetune_epochs'],
            'finetune_lr': optimization['finetune_lr'],
            'model_name': model_name
        })

    print(f'Run LFT analysis...')
    start = time.time()
    results = run_analysis_lft_cnn_head(
        model_name=model_name,
        size_scale=size_scale,
        hidden_dim=hidden_dim,
        train_images=train_images,
        train_labels=train_labels,
        test_images=evaluation_images,
        test_labels=evaluation_labels,
        test_indices=evaluation_indices,
        num_neighbors=optimization['num_neighbors'],
        optimization=optimization,
        model_on_cluster=model_on_cluster,
        use_wandb=use_wandb,
        save_results=True,
    )
    end = time.time()
    print(f'Finished in {end - start} seconds')

    if use_wandb:
        wandb.finish()

    print('\n' + '*' * 80)
    print(f'Accuracy of global model on {results["num_samples"]} samples: {(results["global"]["accuracy"] * 100):.2f}%')
    print(f'Accuracy of locally fine-tuned classifier on {results["num_samples"]} samples: {(results["lft"]["accuracy"] * 100):.2f}%')
    print('*' * 80 + '\n')

    # Prepare results for logging
    results['accuracy_global'] = results['global']['accuracy']
    results['accuracy_lft'] = results['lft']['accuracy']
    results['evaluation_set'] = evaluation_set

    # Save results in log book
    filename = os.path.splitext(os.path.basename(__file__))[0]
    log_file = f'{filename}.csv'
    log_to_csv(results, log_file)

    # Print results
    cols = ['date', 'time', 'accuracy_global', 'accuracy_lft', 'evaluation_set', 'hidden_dim',  'num_neighbors', 'finetune_epochs',
            'finetune_lr', 'num_samples', 'model_name']
    print_tabulated(log_file, cols=cols, head=10)