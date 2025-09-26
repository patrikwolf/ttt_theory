import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import faiss

from scaling.log_book.read_and_write import log_to_csv, print_tabulated
from scaling.models.imagenet import load_imagenet_mlp_head
from scaling.utils.data_loader_imagenet import load_clip_embeddings
from scaling.utils.directory import get_results_dir
from scaling.utils.faiss_knn import batched_faiss_search


def run_analysis_majority_mlp_head(
        model_name,
        train_embeddings,
        train_labels,
        val_embeddings,
        val_labels,
        test_embeddings,
        test_labels,
        num_neighbors,
        hidden_dim,
        dropout_rate,
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
    mlp_head = load_imagenet_mlp_head(
        device=device,
        model_name=model_name,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
        cluster=model_on_cluster,
    )
    mlp_head.to(device)

    # Split model into two parts: the feature extractor and the last linear layer
    feature_extractor = nn.Sequential(
        mlp_head.layer1,
        mlp_head.activation
    )
    last_linear_layer = mlp_head.layer2

    # Map the embeddings to the feature space
    train_features = feature_extractor(train_embeddings.to(device)).detach()
    test_features = feature_extractor(test_embeddings.to(device)).detach()
    val_features = feature_extractor(val_embeddings.to(device)).detach()

    # Build FAISS index
    faiss_index = faiss.IndexFlatL2(train_features.shape[1])
    faiss_index.add(train_features.detach().cpu().numpy())

    print('*' * 80)
    print(f'Evaluating majority vote on the test set on MLP with hidden dimension {hidden_dim} with {num_neighbors} neighbors...')
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
    )

    print('*' * 80)
    print(f'Evaluating majority vote on the validation set on MLP with hidden dimension {hidden_dim} with {num_neighbors} neighbors...')
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
        'model_name': model_name,
        'hidden_dim': hidden_dim,
        'num_neighbors': num_neighbors,
    }

    if save_results:
        results_dir = get_results_dir(experiment_name='imagenet_majority_mlp_heads')
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
        num_classes=1000,
):
    # Start time
    start = time.time()

    # Subset test embeddings and labels
    if test_indices is not None:
        test_embeddings = test_embeddings[test_indices]
        test_labels = test_labels[test_indices]

    # Criterion for loss calculation
    criterion = nn.CrossEntropyLoss(reduction='none')

    # Global predictions of MLP head
    print(f'Compute global predictions...')
    torch_classifier.eval()
    with torch.no_grad():
        # Move test embeddings to device
        test_tensor = test_embeddings.to(device)

        # Forward pass
        logits_global = torch_classifier(test_tensor)

        # Compute CE loss
        ce_loss_list_global = criterion(logits_global, test_labels.to(device))

        # Get predictions
        pred_global = torch.argmax(logits_global, dim=1)

    correct_list_global = (pred_global.cpu() == test_labels)
    correct_global = correct_list_global.sum().item()
    print(f'The script is running for {time.time() - start:.2f} seconds')

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

    # CE loss
    print(f'Computing CE loss...')
    logits_majority = torch.tensor(counts, dtype=torch.float32).to(device)
    labels_tensor = test_labels.to(device)
    ce_loss_list_majority = criterion(logits_majority, labels_tensor)
    print(f'The script is running for {time.time() - start:.2f} seconds')

    # Compute accuracies
    print(f'Computing accuracies...')
    num_samples = len(test_labels)
    accuracy_global = float(correct_global / num_samples)
    accuracy_majority = float(correct_majority / num_samples)

    print(f'Total runtime {time.time() - start:.2f} seconds')

    results = {
        'global': {
            'accuracy': accuracy_global,
            'correct_list': correct_list_global.tolist(),
            'ce_loss_list': ce_loss_list_global.cpu().numpy().tolist(),
        },
        'majority': {
            'accuracy': accuracy_majority,
            'correct_list': correct_list_majority.tolist(),
            'ce_loss_list': ce_loss_list_majority.cpu().numpy().tolist(),
        },
        'num_samples': num_samples,
    }

    return results


if __name__ == '__main__':
    # Fix random seed for reproducibility
    np.random.seed(42)

    # Load the embeddings and labels
    print(f'Loading embeddings and labels...')
    train_embeddings, train_labels = load_clip_embeddings('training_data')
    test_embeddings, test_labels = load_clip_embeddings('test_data')
    val_embeddings, val_labels = load_clip_embeddings('validation_data')

    # Parameters for majority vote analysis
    num_neighbors = 100
    faiss_batch_size = 1          # use batch_size 1 on local machine and 1000 on cluster!

    # Model name
    hidden_dim = 250
    learning_rate = 4 * 1e-4
    batch_size = 450
    dropout_rate = 0.05
    weight_decay = 0.0
    num_epochs = 50
    model_on_cluster = False
    model_name = (f'torch_mlp_classifier_'
                  f'lr{learning_rate:.4f}_'
                  f'bs{batch_size}_'
                  f'hd{hidden_dim}_'
                  f'wd{weight_decay:.4f}_'
                  f'dr{dropout_rate:.2f}_'
                  f'ne{num_epochs}')

    print(f'Run majority vote analysis...')
    results = run_analysis_majority_mlp_head(
        model_name=model_name,
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        val_embeddings=val_embeddings,
        val_labels=val_labels,
        test_embeddings=test_embeddings,
        test_labels=test_labels,
        num_neighbors=num_neighbors,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
        batch_size=faiss_batch_size,
        model_on_cluster=model_on_cluster,
    )

    print('*' * 80)
    print(f'Validation accuracy of global MLP head: {(results["global"]["accuracy_val"] * 100):.2f}%')
    print(f'Validation accuracy of majority vote: {(results["majority"]["accuracy_val"] * 100):.2f}%')
    print('–' * 40)
    print(f'Test accuracy of global MLP head: {(results["global"]["accuracy_test"] * 100):.2f}%')
    print(f'Test accuracy of majority vote: {(results["majority"]["accuracy_test"] * 100):.2f}%')
    print('*' * 80)

    # Prepare results for logging
    results['accuracy_global_val'] = results['global']['accuracy_val']
    results['accuracy_majority_val'] = results['majority']['accuracy_val']
    results['accuracy_global_test'] = results['global']['accuracy_test']
    results['accuracy_majority_test'] = results['majority']['accuracy_test']
    results['num_samples_val'] = len(results['global']['correct_list_val'])
    results['num_samples_test'] = len(results['global']['correct_list_test'])

    # Save results in log book
    filename = os.path.splitext(os.path.basename(__file__))[0]
    log_file = f'{filename}.csv'
    log_to_csv(results, log_file)

    # Print results
    cols = ['date', 'time', 'accuracy_global_val', 'accuracy_majority_val', 'accuracy_global_test',
            'accuracy_majority_test', 'hidden_dim', 'num_neighbors', 'num_samples_val', 'num_samples_test', 'model_name']
    all_cols, df = print_tabulated(log_file, cols=cols, head=10)