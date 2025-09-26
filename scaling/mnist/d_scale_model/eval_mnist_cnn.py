import os
import json
import torch
import numpy as np

from scaling.log_book.read_and_write import log_to_csv, print_tabulated
from scaling.mnist.parameters.param_helper import get_optimal_mnist_cnn_parameters
from scaling.models.evaluation import evaluate_model
from scaling.models.mnist import load_global_mnist_model
from scaling.utils.data_loader_mnist import load_mnist_embeddings
from scaling.utils.directory import get_results_dir

def evaluate_and_save_results(models):
    # Fix random seed for reproducibility
    np.random.seed(42)

    # Load the embeddings and labels
    print(f'Loading embeddings and labels...')
    test_images, _, test_labels, _ = load_mnist_embeddings('test_data')
    val_images, _, val_labels, _ = load_mnist_embeddings('validation_data')

    results_dict = {}

    # Log file
    filename = os.path.splitext(os.path.basename(__file__))[0]
    log_file = f'{filename}.csv'

    # Iterate over the hidden dimensions
    for hidden_dim, params in models.items():
        model_name = (f'mnist_scaled_model_'
                      f'hd{hidden_dim}_'
                      f'ss{params["size_scale"]}_'
                      f'ot-{params["optimization_type"]}_'
                      f'lr{params["learning_rate"]}'
                      f'bs{params["batch_size"]}_'
                      f'ne{params["num_epochs"]}')
        print(f'Run CNN head analysis with hidden dimension {hidden_dim}...')

        # Run the analysis on the test set
        results_test = run_analysis_mnist_scaled_cnn(
            model_name=model_name,
            size_scale=params['size_scale'],
            test_images=test_images,
            test_labels=test_labels,
            hidden_dim=int(hidden_dim),
            cluster=params['cluster'],
        )
        print(f'CNN accuracy on TEST set with hidden dimension {hidden_dim}: '
              f'{(results_test["accuracy_cnn"] * 100):.2f}%')
        print('*' * 80)

        # Run the analysis on the validation set
        results_val = run_analysis_mnist_scaled_cnn(
            model_name=model_name,
            size_scale=params['size_scale'],
            test_images=val_images,
            test_labels=val_labels,
            hidden_dim=int(hidden_dim),
            cluster=params['cluster'],
        )
        print(f'CNN accuracy on VALIDATION set with hidden dimension {hidden_dim}: '
              f'{(results_val["accuracy_cnn"] * 100):.2f}%')
        print('-' * 40)

        # Add parameters to the results
        results_test['accuracy_cnn_test'] = results_test['accuracy_cnn']
        results_test['accuracy_cnn_val'] = results_val['accuracy_cnn']
        results_test['learning_rate'] = params['learning_rate']
        results_test['batch_size'] = params['batch_size']
        results_test['num_epochs'] = params['num_epochs']
        results_test['size_scale'] = params['size_scale']
        results_test['optimization_type'] = params['optimization_type']
        results_test['total_params'] = results_test['total_params']

        # Store the results in the model dictionary
        results_dict[hidden_dim] = results_test

        # Save results in log book
        log_to_csv(results_test, log_file)

    # Save the results
    results_dir = get_results_dir(experiment_name='mnist_eval_scaled_classifier')
    with open(f'{results_dir}/results.json', 'w') as f:
        json.dump(results_dict, f)

    print(f'Combined results saved to {results_dir}/results.json')

    # Print results
    cols = ['date', 'time', 'accuracy_cnn_val', 'accuracy_cnn_test', 'hidden_dim', 'learning_rate', 'batch_size', 'model_name', 'num_samples']
    print_tabulated(log_file, cols=cols, head=10)

    print(f'\n--> Combined results saved to {results_dir}/results.json')
    return results_dir.name


def run_analysis_mnist_scaled_cnn(
        model_name,
        size_scale,
        test_images,
        test_labels,
        hidden_dim,
        cluster=False,
        test_indices=None,
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
        cluster=cluster,
        only_head=False,
    )

    # Number of parameters
    total_params = sum(p.numel() for p in scaled_cnn.parameters())

    # Filter test images and labels
    if test_indices is not None:
        test_images = test_images[test_indices]
        test_labels = test_labels[test_indices]
        num_samples = len(test_indices)
    else:
        num_samples = len(test_labels)

    print(f'Evaluating CNN with hidden dim {hidden_dim}...')
    accuracy_cnn, correct_list, ce_loss_list = evaluate_model(
        model=scaled_cnn,
        test_embeddings=test_images,
        test_labels=test_labels,
    )

    # Save the results in the results_dict
    results_dict = {
        'accuracy_cnn': accuracy_cnn,
        'correct_list': correct_list.tolist(),
        'ce_loss_list': ce_loss_list.tolist(),
        'model_name': model_name,
        'hidden_dim': hidden_dim,
        'num_samples': num_samples,
        'total_params': total_params,
    }

    if save_results:
        results_dir = get_results_dir(experiment_name='mnist_cnn_eval')
        results_file = f'{results_dir}/results.json'
        with open(results_file, 'w') as f:
            json.dump(results_dict, f)
        print(f'Results saved to {results_file}')
    else:
        print(f'---> Result saving within evaluation function is disabled.')

    return results_dict


if __name__ == '__main__':
    # Parameters for the analysis
    models = get_optimal_mnist_cnn_parameters()

    # Evaluate and save results
    timestamp = evaluate_and_save_results(models)
    print(timestamp)