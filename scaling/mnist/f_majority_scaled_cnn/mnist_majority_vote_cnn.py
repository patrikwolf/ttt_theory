import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import faiss

from scaling.models.evaluation import evaluate_model
from scaling.log_book.read_and_write import log_to_csv, print_tabulated
from scaling.mnist.parameters.param_helper import get_optimal_mnist_cnn_parameters
from scaling.models.mnist import load_global_mnist_model
from scaling.utils.data_loader_mnist import load_mnist_embeddings
from scaling.utils.directory import get_results_dir
from scaling.utils.faiss_knn import batched_faiss_search

def run_analysis_mnist_majority_cnn(
        model_name,
        size_scale,
        train_images,
        train_labels,
        val_images,
        val_labels,
        test_images,
        test_labels,
        num_neighbors,
        hidden_dim,
        model_on_cluster,
        val_indices=None,
        test_indices=None,
        batch_size=1,
        save_results=True,
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
    val_features = scaled_cnn(val_images.to(device), extract_last_layer=True).detach()

    # Get the last linear layer
    last_linear_layer = scaled_cnn.classifier

    # Build FAISS index
    faiss_index = faiss.IndexFlatL2(train_features.shape[1])
    faiss_index.add(train_features.detach().cpu().numpy())

    print('*' * 80)
    print(f'Evaluating majority vote on the test set on CNN with hidden dimension {hidden_dim} with {num_neighbors} neighbors...')
    print('*' * 80)
    results_test = evaluate_majority_vote_vectorized(
        torch_classifier=last_linear_layer,
        device=device,
        faiss_index=faiss_index,
        train_labels=train_labels,
        test_embeddings=test_features,
        test_labels=test_labels,
        test_indices=test_indices,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        num_classes=10,
    )

    print('*' * 80)
    print(f'Evaluating majority vote on the validation set on CNN with hidden dimension {hidden_dim} with {num_neighbors} neighbors...')
    print('*' * 80)
    results_val = evaluate_majority_vote_vectorized(
        torch_classifier=last_linear_layer,
        device=device,
        faiss_index=faiss_index,
        train_labels=train_labels,
        test_embeddings=val_features,
        test_labels=val_labels,
        test_indices=val_indices,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        num_classes=10,
    )

    # Save the results in the results_dict
    results_dict = {
        'global': {
            'accuracy_test': results_test['global']['accuracy'],
            'accuracy_val': results_val['global']['accuracy'],
            'correct_list_test': results_test['global']['correct_list'],
            'correct_list_val': results_val['global']['correct_list'],
            'ce_loss_list_test': results_test['global']['ce_loss_list'],
            'ce_loss_list_val': results_val['global']['ce_loss_list'],
        },
        'majority': {
            'accuracy_test': results_test['majority']['accuracy'],
            'accuracy_val': results_val['majority']['accuracy'],
            'correct_list_test': results_test['majority']['correct_list'],
            'correct_list_val': results_val['majority']['correct_list'],
            'ce_loss_list_test': results_test['majority']['ce_loss_list'],
            'ce_loss_list_val': results_val['majority']['ce_loss_list'],
        },
        'num_samples_test': results_test['num_samples'],
        'num_samples_val': results_val['num_samples'],
        'model_name': model_name,
        'hidden_dim': hidden_dim,
        'total_params': total_params,
        'num_neighbors': num_neighbors,
    }

    if save_results:
        results_dir = get_results_dir(experiment_name='mnist_majority_cnn')
        results_file = f'{results_dir}/results.json'
        with open(results_file, 'w') as f:
            json.dump(results_dict, f)
        print(f'Results saved to {results_file}')
    else:
        print(f'---> Result saving within evaluation function is disabled.')

    return results_dict


def evaluate_majority_vote_vectorized(
        torch_classifier,
        device,
        faiss_index,
        train_labels,
        test_embeddings,
        test_labels,
        test_indices,
        num_neighbors,
        batch_size,
        num_classes=1000
):
    # Start time
    start = time.time()

    # Subset test embeddings and labels
    if test_indices is not None:
        test_embeddings = test_embeddings[test_indices]
        test_labels = test_labels[test_indices]

    # Evaluate the global model
    accuracy_global, correct_list_global, ce_loss_global = evaluate_model(
        model=torch_classifier,
        test_embeddings=test_embeddings,
        test_labels=test_labels,
    )

    # Move test embeddings to CPU for FAISS search
    test_embeddings = test_embeddings.cpu()

    # FAISS k-NN search (in batches)
    print(f'Performing batched FAISS k-NN search with batch size {batch_size}...')
    neighbor_indices = batched_faiss_search(faiss_index, test_embeddings, num_neighbors, batch_size=batch_size)
    neighbor_labels = train_labels[neighbor_indices]  # shape: (num_samples, num_neighbors)
    print(f'The script is running for {time.time() - start:.2f} seconds')

    # Majority vote
    print(f'Performing majority voting...')
    counts = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=num_classes),
        axis=1,
        arr=neighbor_labels
    )
    most_common_labels = np.argmax(counts, axis=1)
    correct_list_majority = (most_common_labels == test_labels.numpy())
    correct_majority = correct_list_majority.sum()
    print(f'The script is running for {time.time() - start:.2f} seconds')

    # Criterion for loss calculation
    criterion = nn.CrossEntropyLoss(reduction='none')

    # CE loss
    print(f'Computing CE loss...')
    logits_majority = torch.tensor(counts, dtype=torch.float32).to(device)
    labels_tensor = test_labels.to(device)
    ce_loss_list_majority = criterion(logits_majority, labels_tensor)
    print(f'Total runtime {time.time() - start:.2f} seconds')

    results = {
        'global': {
            'accuracy': accuracy_global,
            'correct_list': correct_list_global.tolist(),
            'ce_loss_list': ce_loss_global.tolist(),
        },
        'majority': {
            'accuracy': float(correct_majority / len(test_labels)),
            'correct_list': correct_list_majority.tolist(),
            'ce_loss_list': ce_loss_list_majority.cpu().numpy().tolist(),
        },
        'num_samples': len(test_labels),
    }

    return results


if __name__ == '__main__':
    # Fix random seed for reproducibility
    np.random.seed(42)

    # Load the embeddings and labels
    print(f'Loading embeddings and labels...')
    train_images, _, train_labels, _ = load_mnist_embeddings('training_data')
    test_images, _, test_labels, _ = load_mnist_embeddings('test_data')
    val_images, _, val_labels, _ = load_mnist_embeddings('validation_data')

    # Parameters for majority vote analysis
    num_neighbors = 100
    faiss_batch_size = 1          # use batch_size 1 on local machine and 1000 on cluster!

    # Model name
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

    print(f'Run majority vote analysis...')
    results = run_analysis_mnist_majority_cnn(
        model_name=model_name,
        size_scale=size_scale,
        train_images=train_images,
        train_labels=train_labels,
        val_images=val_images,
        val_labels=val_labels,
        test_images=test_images,
        test_labels=test_labels,
        num_neighbors=num_neighbors,
        hidden_dim=hidden_dim,
        batch_size=faiss_batch_size,
        model_on_cluster=model_on_cluster,
        save_results=True
    )

    print('*' * 80)
    print(f'Validation accuracy of global CNN: {(results["global"]["accuracy_val"] * 100):.2f}%')
    print(f'Validation accuracy of majority vote: {(results["majority"]["accuracy_val"] * 100):.2f}%')
    print('–' * 40)
    print(f'Test accuracy of global CNN: {(results["global"]["accuracy_test"] * 100):.2f}%')
    print(f'Test accuracy of majority vote: {(results["majority"]["accuracy_test"] * 100):.2f}%')
    print('*' * 80)

    # Prepare results for logging
    results['accuracy_global_val'] = results['global']['accuracy_val']
    results['accuracy_majority_val'] = results['majority']['accuracy_val']
    results['accuracy_global_test'] = results['global']['accuracy_test']
    results['accuracy_majority_test'] = results['majority']['accuracy_test']

    # Save results in log book
    filename = os.path.splitext(os.path.basename(__file__))[0]
    log_file = f'{filename}.csv'
    log_to_csv(results, log_file)

    # Print results
    cols = ['date', 'time', 'accuracy_global_val', 'accuracy_majority_val', 'accuracy_global_test',
            'accuracy_majority_test', 'hidden_dim', 'total_params', 'num_neighbors', 'num_samples_val', 'num_samples_test', 'model_name']
    all_cols, df = print_tabulated(log_file, cols=cols, head=10)