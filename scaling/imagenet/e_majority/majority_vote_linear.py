import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import faiss

from scaling.imagenet.parameters.param_helper import get_name_of_global_linear_model
from scaling.log_book.read_and_write import log_to_csv, print_tabulated
from scaling.models.imagenet import load_linear_imagenet_model
from scaling.utils.data_loader import load_evaluation_set
from scaling.utils.data_loader_imagenet import load_clip_embeddings
from scaling.utils.directory import get_results_dir
from scaling.utils.faiss_knn import batched_faiss_search


def run_analysis_majority_vote(
        model_name,
        train_embeddings,
        train_labels,
        test_embeddings,
        test_labels,
        test_indices,
        num_neighbors,
        batch_size=1,
        model_on_cluster=False,
        save_results=True,
):
    # Device configuration (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load the model
    global_model = load_linear_imagenet_model(device, model_name=model_name, cluster=model_on_cluster)
    global_model.to(device)

    # Build FAISS index
    faiss_index = faiss.IndexFlatL2(train_embeddings.shape[1])
    faiss_index.add(train_embeddings)

    print('Evaluating majority vote model...')
    results = evaluate_majority_vote_vectorized(
        torch_classifier=global_model,
        device=device,
        faiss_index=faiss_index,
        train_labels=train_labels,
        test_embeddings=test_embeddings,
        test_labels=test_labels,
        test_indices=test_indices,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
    )

    # Save the results in the results_dict
    results['model_name'] = model_name
    results['num_neighbors'] = num_neighbors
    results['num_samples'] = len(test_indices)

    if save_results:
        results_dir = get_results_dir(experiment_name='imagenet_majority_vote')
        results_file = f'{results_dir}/results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f)
        print(f'Results saved to {results_file}')
    else:
        print(f'---> Result saving within evaluation function is disabled.')

    return results


def evaluate_majority_vote(
        torch_classifier,
        device,
        faiss_index,
        train_labels,
        test_embeddings,
        test_labels,
        test_indices,
        num_neighbors,
        num_classes=1000
):
    # Start time
    start = time.time()

    # Initialize counters for correct predictions
    num_samples = len(test_indices)
    correct_global = 0
    correct_majority = 0
    correct_majority_list = np.zeros(num_samples, dtype=int)
    ce_loss_list = np.zeros(num_samples)

    for idx, test_idx in enumerate(test_indices):
        if (idx + 1) % 100 == 0:
            print(f'Evaluating sample {idx + 1}/{num_samples} ({100 * (idx + 1) / num_samples:.2f}%)')

        # Get the test sample embedding and its true label
        test_sample_embedding = test_embeddings[test_idx].reshape(1, -1)
        true_label = test_labels[test_idx]

        # Get a baseline prediction from the global model
        torch_classifier.eval()
        with torch.no_grad():
            test_tensor = test_sample_embedding.to(device)
            logits = torch_classifier(test_tensor)
            _, prediction_global = torch.max(logits, 1)

        if prediction_global.item() == true_label:
            correct_global += 1

        # Fetch the k nearest neighbors
        _, indices = faiss_index.search(test_sample_embedding, num_neighbors)
        neighbor_indices = indices[0]
        neighbor_labels = train_labels[neighbor_indices]

        # Majority vote
        counts = np.bincount(neighbor_labels, minlength=num_classes)
        labels = np.arange(num_classes)

        #labels, counts = np.unique(neighbor_labels, return_counts=True)
        most_common_label = labels[np.argmax(counts)]

        if most_common_label == true_label:
            correct_majority += 1
            correct_majority_list[idx] = 1

        # CE loss (of the label frequency distribution)
        criterion = nn.CrossEntropyLoss()
        logits = torch.tensor(counts).float().unsqueeze(0).to(device)
        true_label = true_label.unsqueeze(0).to(device)
        ce_loss = criterion(logits, true_label)
        ce_loss_list[idx] = ce_loss.item()

        if (idx + 1) % 100 == 0:
            print(f'Current accuracy of global classifier: {(100 * correct_global / (idx + 1)):.2f}')
            print(f'Current accuracy of majority vote: {(100 * correct_majority / (idx + 1)):.2f}')
            print('*' * 100)

    # Calculate the accuracy for both global and local fine-tuning
    accuracy_global = correct_global / num_samples
    accuracy_majority = correct_majority / num_samples

    print(f'Total runtime {time.time() - start:.2f} seconds')

    return accuracy_global, accuracy_majority, correct_majority_list, ce_loss_list


def evaluate_majority_vote_vectorized(
        torch_classifier,
        device,
        faiss_index,
        train_labels,
        test_embeddings,
        test_labels,
        test_indices,
        num_neighbors,
        batch_size
):
    # Start time
    start = time.time()

    # Subset test embeddings and labels
    test_embeddings = test_embeddings[test_indices]
    test_labels = test_labels[test_indices]

    # Initialize criterion for CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(reduction='none')

    # Global predictions (vectorized)
    print(f'Compute global predictions...')
    torch_classifier.eval()
    with torch.no_grad():
        # Move tensors to the appropriate device
        test_tensor = test_embeddings.to(device)
        labels_tensor = test_labels.to(device)

        # Forward pass
        logits_global = torch_classifier(test_tensor)

        # Compute CrossEntropy loss
        ce_loss_list_global = criterion(logits_global, labels_tensor)

        # Get the predicted class
        pred_global = torch.argmax(logits_global, dim=1)

    correct_list_global = (pred_global.cpu() == test_labels)
    correct_global = correct_list_global.sum().item()
    print(f'The script is running for {time.time() - start:.2f} seconds')

    # FAISS k-NN search (batch)
    print(f'Performing batched FAISS k-NN search with batch size {batch_size}...')
    neighbor_indices = batched_faiss_search(faiss_index, test_embeddings, num_neighbors, batch_size=batch_size)
    neighbor_labels = train_labels[neighbor_indices]
    print(f'The script is running for {time.time() - start:.2f} seconds')

    # Majority vote (vectorized with bincount)
    print(f'Performing majority voting...')
    num_classes = 1000
    counts = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=num_classes),
        axis=1,
        arr=neighbor_labels
    )
    most_common_labels = np.argmax(counts, axis=1)
    correct_list_majority = (most_common_labels == test_labels.numpy())
    correct_majority = correct_list_majority.sum()
    print(f'The script is running for {time.time() - start:.2f} seconds')

    # CE loss (vectorized)
    print(f'Computing CE loss...')
    logits_majority = torch.tensor(counts, dtype=torch.float32).to(device)
    labels_tensor = test_labels.to(device)
    ce_loss_list_majority = criterion(logits_majority, labels_tensor)
    print(f'The script is running for {time.time() - start:.2f} seconds')

    # Compute accuracies
    print(f'Computing accuracies...')
    num_samples = len(test_indices)
    accuracy_global = float(correct_global / num_samples)
    accuracy_majority = float(correct_majority / num_samples)

    print(f'Total runtime {time.time() - start:.2f} seconds')

    # Add results to dictionary
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

    # Load the evaluation set
    evaluation_embeddings, evaluation_labels, evaluation_indices = load_evaluation_set(
        test_embeddings,
        test_labels,
        val_embeddings,
        val_labels,
        evaluation_set='test',
        num_indices_per_class='all'
    )

    # Parameters for majority vote analysis
    num_neighbors = 100
    batch_size = 1               # use batch_size 1 on local machine and 1000 on cluster!

    # Model name
    model_name, model_on_cluster = get_name_of_global_linear_model()

    print(f'Run majority vote analysis...')
    results = run_analysis_majority_vote(
        model_name=model_name,
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        test_embeddings=evaluation_embeddings,
        test_labels=evaluation_labels,
        test_indices=evaluation_indices,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        model_on_cluster=model_on_cluster,
    )

    print(f'Accuracy of global model on {len(evaluation_indices)} samples: {(results["global"]["accuracy"] * 100):.2f}%')
    print(f'Accuracy of majority vote on {len(evaluation_indices)} samples: {(results["majority"]["accuracy"] * 100):.2f}%')

    # Prepare results for logging
    results['accuracy_global'] = results['global']['accuracy']
    results['accuracy_majority'] = results['majority']['accuracy']

    # Save results in log book
    filename = os.path.splitext(os.path.basename(__file__))[0]
    log_file = f'{filename}.csv'
    log_to_csv(results, log_file)

    # Print results
    cols = ['date', 'time', 'accuracy_global', 'accuracy_majority', 'num_neighbors', 'num_samples', 'model_name']
    all_cols, df = print_tabulated(log_file, head=10, cols=cols, sort_by='accuracy_majority', ascending=False, cluster=False)