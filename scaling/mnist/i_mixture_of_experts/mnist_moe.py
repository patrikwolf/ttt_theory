import copy
import time
import json
import torch
import faiss
import argparse
import itertools
import numpy as np
import torch.nn as nn
import torch.optim as optim

from scipy.cluster.vq import kmeans, vq
from scaling.mnist.parameters.param_helper import get_optimal_mnist_cnn_parameters
from scaling.mnist.h_lft_global_eval.lft_global_eval import locally_fine_tune_torch_model
from scaling.mnist.parameters.param_helper import get_optimal_moe_parameters
from scaling.models.evaluation import evaluate_model
from scaling.models.mnist import load_global_mnist_model
from scaling.utils.data_loader import load_evaluation_set
from scaling.utils.data_loader_mnist import load_mnist_embeddings
from scaling.utils.directory import get_results_dir

def run_analysis_moe(
        model_name,
        size_scale,
        num_experts,
        train_images,
        train_labels,
        test_images,
        test_labels,
        test_indices,
        optimization,
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
    test_labels = test_labels.to(device)

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

    # Train MoE on each cluster
    print('Training an expert on each cluster centroid...')
    experts = train_experts(
        global_model=last_linear_layer,
        train_features=train_features,
        train_labels=train_labels,
        num_experts=num_experts,
        opt_params=optimization,
        device=device,
    )

    # Evaluate the MoE model on the test set
    print('Evaluating the MoE model on the test set...')
    results_moe = evaluate_mixture_of_experts(
        experts=experts,
        test_features=test_features,
        test_labels=test_labels,
        test_indices=test_indices,
        device=device,
    )
    results['moe'] = results_moe

    # Add parameters to results dictionary
    results['model_name'] = model_name
    results['total_params'] = total_params
    results['num_samples'] = len(test_indices)
    results['num_experts'] = num_experts
    results['optimization'] = optimization

    return results


def cluster_dataset(
        train_features,
        train_labels,
        num_experts,
):
    # Splitting the training set into clusters using K-means on the last-layer features
    print(f'Splitting training set of size {len(train_labels)} into {num_experts} clusters')
    centroids = kmeans(train_features.cpu().numpy(), num_experts, iter=20)[0]

    # Compute the cluster sizes
    cluster_assignment = vq(train_features.cpu().numpy(), centroids)[0].astype(int)
    cluster_counts = np.bincount(cluster_assignment, minlength=num_experts)

    # Print cluster size statistics
    mean_size = np.mean(cluster_counts)
    min_size = np.min(cluster_counts)
    max_size = np.max(cluster_counts)
    std_dev_size = np.std(cluster_counts)
    print('*' * 50)
    print(f'Cluster sizes: Count: {num_experts}, Mean={mean_size:.2f}, Min={min_size}, Max={max_size}, StdDev={std_dev_size:.2f}')

    return centroids


def train_experts(
        global_model,
        train_features,
        train_labels,
        num_experts,
        opt_params,
        device,
):
    # Initialize list of experts
    experts = []
    num_neighbors = opt_params['num_neighbors']

    # Build FAISS index
    faiss_index = faiss.IndexFlatL2(train_features.shape[1])
    faiss_index.add(train_features.detach().cpu().numpy())

    # Cluster the training set
    centroids = cluster_dataset(
        train_features=train_features,
        train_labels=train_labels,
        num_experts=num_experts
    )

    # Iterate over centroids and train an expert for each
    for idx, centroid in enumerate(centroids):
        if (idx + 1) % 500 == 0:
            print(f'Training expert on centroid {idx + 1} out of {len(centroids)} ({100 * (idx + 1) / len(centroids):.2f}%)')

        # Get the centroid feature
        centroid_feature = centroid.reshape(1, -1)

        # Fetch the k nearest neighbors
        _, indices = faiss_index.search(centroid_feature, num_neighbors)
        neighbor_indices = indices[0]
        neighbor_features = train_features[neighbor_indices]
        neighbor_labels = train_labels[neighbor_indices]

        # Local fine-tuning
        if opt_params['type'] == 'LBFGS':
            local_model = locally_fine_tune_torch_model_lbfgs(
                linear_model=global_model,
                neighbor_features=neighbor_features,
                neighbor_labels=neighbor_labels,
                device=device,
                optimization_params=opt_params,
            )
        elif opt_params['type'] == 'ADAM':
            local_model = locally_fine_tune_torch_model(
                global_model=global_model,
                neighbor_embeddings=neighbor_features,
                neighbor_labels=neighbor_labels,
                device=device,
                optimization_params=opt_params,
            )
        else:
            raise ValueError(f'Unknown optimization type: {opt_params["type"]}')

        # Append to list of experts
        experts.append({
            'linear_head': local_model,
            'centroid_feature': centroid_feature,
        })

    return experts


def locally_fine_tune_torch_model_lbfgs(
        linear_model,
        neighbor_features,
        neighbor_labels,
        device,
        optimization_params,
):
    # Create a deep copy of the global model to fine-tune.
    local_linear_head = copy.deepcopy(linear_model).to(device)

    # Copy weights and bias
    global_weights = linear_model.weight.clone().detach().cpu()
    global_bias = linear_model.bias.clone().detach().cpu()

    # Set optimizer and loss function
    optimizer = optim.LBFGS(local_linear_head.parameters())
    criterion = nn.CrossEntropyLoss()

    # Set the model to training mode
    local_linear_head.train()

    for _ in range(10):

        def closure():
            optimizer.zero_grad()
            outputs = local_linear_head(neighbor_features)

            ce_loss = criterion(outputs, neighbor_labels)

            weight_difference = local_linear_head.weight - global_weights
            reg_loss = torch.mean(weight_difference ** 2)
            if global_bias is not None:
                bias_difference = local_linear_head.bias - global_bias
                reg_loss += torch.mean(bias_difference ** 2)

            total_loss = ce_loss + optimization_params['weight_decay'] * reg_loss
            total_loss.backward()

            return total_loss

        optimizer.step(closure)

    return local_linear_head


def evaluate_mixture_of_experts(
        experts,
        test_features,
        test_labels,
        test_indices,
        device,
):
    # Initialize results list
    num_samples = len(test_indices)
    correct_list = np.zeros(num_samples, dtype=int)
    ce_loss_list = np.zeros(num_samples, dtype=float)

    # Extract list of centroid features
    centroid_features = np.squeeze(np.array([expert['centroid_feature'] for expert in experts]))

    # Iterate over test samples
    start_eval = time.time()
    for idx, test_idx in enumerate(test_indices):
        if (idx + 1) % 500 == 0:
            print(f'Evaluating sample {idx + 1}/{num_samples} ({100 * (idx + 1) / num_samples:.2f}%) after '
                  f'{((time.time() - start_eval) / 60):.2f} minutes...')

        # Get the test sample embedding and its true label
        test_feature = test_features[test_idx].reshape(1, -1)
        true_label = test_labels[test_idx]
        test_label = true_label.to(device).unsqueeze(0)

        # Get the closest expert
        distances = np.linalg.norm(centroid_features - test_feature.cpu().numpy(), axis=1)
        closest_expert_idx = np.argmin(distances)
        closest_expert = experts[closest_expert_idx]['linear_head']
        closest_expert.eval()

        # Evaluate the model on the test feature
        _, correct_moe, ce_moe = evaluate_model(
            model=closest_expert,
            test_embeddings=test_feature,
            test_labels=test_label,
        )

        # Add results to lists
        correct_list[idx] = correct_moe.item()
        ce_loss_list[idx] = ce_moe.item()

    # Compute accuracy
    accuracy = np.sum(correct_list) / num_samples

    results_dict = {
        'type': 'moe',
        'accuracy': float(accuracy),
        'correct_list': correct_list.tolist(),
        'ce_loss_list': ce_loss_list.tolist(),
    }

    return results_dict


if __name__ == '__main__':
    print('*' * 80)
    print('Running MoE model based on the last linear layer of the scaled CNN models.')
    print('*' * 80)

    # Fix random seed for reproducibility
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_id", type=int, required=True)
    parser.add_argument("--datetime", type=str, required=True)
    args = parser.parse_args()

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
    hidden_dim = 192
    size_scale = 1.0
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

    # Prepare for shards
    count = 0
    optimization = {}

    # Parameters for MoE
    num_experts_list = [10]  #[1, 10, 30, 50, 100, 300, 500, 1000, 3000,

    # Parameters for local fine-tuning
    num_neighbors = [20, 30, 40, 50]
    finetune_epochs = [20, 30, 40, 50, 60]
    finetune_lr = [8e-4, 6e-4, 4e-4, 2e-4]

    for ne, nn, epochs, lr in itertools.product(num_experts_list, num_neighbors, finetune_epochs, finetune_lr):
        optimization[f'shard_{count}'] = {
            'type': 'ADAM',
            'num_neighbors': nn,
            'finetune_epochs': epochs,
            'finetune_lr': lr,
            'weight_decay': 0.01,
            'num_experts': ne,
        }
        count += 1

    # Select hyperparameters for the current shard
    shard = f'shard_{args.shard_id}'
    num_experts = optimization[shard]['num_experts']

    overwrite_params = False
    if overwrite_params:
        # Get optimal parameters
        models = get_optimal_moe_parameters()
        model_params = models[str(num_experts)]

        # Overwrite learning rate and epochs
        optimization[shard]['finetune_lr'] = model_params['finetune_lr']
        optimization[shard]['finetune_epochs'] = model_params['finetune_epochs']
        optimization[shard]['num_neighbors'] = model_params['num_neighbors']

        print('\n' + '-' * 80)
        print(f'Overwriting optimization parameters:\n'
              f'num_neighbors = {model_params["num_neighbors"]},\n'
              f'finetune_lr = {model_params["finetune_lr"]},\n'
              f'finetune_epochs = {model_params["finetune_epochs"]}')
        print('-' * 80 + '\n')

    print(f'Run MoE analysis with {num_experts} experts...')
    start = time.time()
    results = run_analysis_moe(
        model_name=model_name,
        size_scale=size_scale,
        num_experts=num_experts,
        train_images=train_images,
        train_labels=train_labels,
        test_images=evaluation_images,
        test_labels=evaluation_labels,
        test_indices=evaluation_indices,
        optimization=optimization[shard],
        model_on_cluster=model_on_cluster,
    )
    end = time.time()
    print(f'Finished in {end - start} seconds')

    print('\n' + '*' * 80)
    print(f'Accuracy of global model on {results["num_samples"]} samples: {(results["global"]["accuracy"] * 100):.2f}%')
    print(f'Accuracy of MoE classifier on {results["num_samples"]} samples: {(results["moe"]["accuracy"] * 100):.2f}%')
    print(f'Number of experts: {num_experts}')
    print('*' * 80 + '\n')

    # Add parameters to results
    results['evaluation_set'] = evaluation_set
    results['hidden_dim'] = hidden_dim
    results['shard_id'] = args.shard_id
    results['datetime'] = args.datetime

    # Save results
    results_dir = get_results_dir(experiment_name='mnist_moe_parallel', timestamp=args.datetime)
    results_file = f'{results_dir}/shard_{args.shard_id}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f)
    print(f'Results saved to {results_file}')
