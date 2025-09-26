import os
import json
import torch
import numpy as np

from scaling.log_book.read_and_write import log_to_csv, print_tabulated
from scaling.models.evaluation import evaluate_model
from scaling.models.imagenet import load_imagenet_mlp_head
from scaling.utils.data_loader import load_evaluation_set
from scaling.utils.data_loader_imagenet import load_clip_embeddings
from scaling.utils.directory import get_results_dir


def run_analysis_mlp_head(
        model_name,
        test_embeddings,
        test_labels,
        hidden_dim,
        dropout_rate,
        cluster=False,
        test_indices=None,
        save_results=True,
):
    # Device configuration (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load the model
    mlp_head = load_imagenet_mlp_head(
        device=device,
        hidden_dim=hidden_dim,
        model_name=model_name,
        dropout_rate=dropout_rate,
        cluster=cluster,
    )

    # Filter test embeddings and labels
    if test_indices is not None:
        test_embeddings = test_embeddings[test_indices]
        test_labels = test_labels[test_indices]
        num_samples = len(test_indices)
    else:
        num_samples = len(test_labels)

    print(f'Evaluating MLP head with hidden dim {hidden_dim}...')
    accuracy_mlp, correct_mlp_list, ce_loss_list = evaluate_model(
        model=mlp_head,
        test_embeddings=test_embeddings,
        test_labels=test_labels,
    )

    # Save the results in the results_dict
    results_dict = {
        'accuracy_mlp': accuracy_mlp,
        'correct_mlp_list': correct_mlp_list.tolist(),
        'ce_loss_list': ce_loss_list.tolist(),
        'model_name': model_name,
        'hidden_dim': hidden_dim,
        'num_samples': num_samples,
    }

    if save_results:
        results_dir = get_results_dir(experiment_name='imagenet_mlp_heads')
        results_file = f'{results_dir}/results.json'
        with open(results_file, 'w') as f:
            json.dump(results_dict, f)
        print(f'Results saved to {results_file}')
    else:
        print(f'---> Result saving within evaluation function is disabled.')

    return results_dict


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

    # Parameters for the analysis
    hidden_dims = [250]
    results_dict = {}
    learning_rate = 4 * 1e-4
    batch_size = 450
    dropout_rate = 0.05
    weight_decay = 0.0
    num_epochs = 50

    # Log file
    filename = os.path.splitext(os.path.basename(__file__))[0]
    log_file = f'{filename}.csv'

    # Iterate over the hidden dimensions
    for hidden_dim in hidden_dims:
        model_name = (f'torch_mlp_classifier_lr{learning_rate:.4f}_bs{batch_size}_hd{hidden_dim}_wd{weight_decay:.4f}'
                      f'_dr{dropout_rate:.2f}_ne{num_epochs}')
        print(f'Run MLP head analysis with hidden dimension {hidden_dim}...')

        # Run the analysis
        results = run_analysis_mlp_head(
            model_name=model_name,
            test_embeddings=evaluation_embeddings,
            test_labels=evaluation_labels,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
        )
        print(f'MLP head accuracy with hidden dimension {hidden_dim}: {(results["accuracy_mlp"] * 100):.2f}%')
        print('*' * 80)

        # Add parameters to the results
        results['learning_rate'] = learning_rate
        results['batch_size'] = batch_size

        # Store the results in the model dictionary
        results_dict[hidden_dim] = results

        # Save results in log book
        log_to_csv(results, log_file)

    # Save the results
    results_dir = get_results_dir(experiment_name='imagenet_mlp_heads')
    with open(f'{results_dir}/results.json', 'w') as f:
        json.dump(results_dict, f)

    print(f'--> Combined results saved to {results_dir}/results.json')
    print(f'Evaluation on the {evaluation_set} set.')

    # Print results
    cols = ['date', 'time', 'hidden_dim', 'learning_rate', 'batch_size', 'model_name', 'num_samples', 'accuracy_mlp']
    all_cols, df = print_tabulated(log_file, cols=cols, head=10)