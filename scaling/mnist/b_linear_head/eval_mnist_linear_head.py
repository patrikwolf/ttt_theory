import os
import json
import numpy as np
import torch

from scaling.log_book.read_and_write import log_to_csv, print_tabulated
from scaling.mnist.parameters.param_helper import get_name_of_mnist_linear_head
from scaling.models.mnist import load_linear_mnist_head
from scaling.utils.data_loader import load_evaluation_set
from scaling.utils.data_loader_mnist import load_mnist_embeddings
from scaling.utils.directory import get_results_dir
from scaling.models.evaluation import evaluate_model


def run_analysis_mnist_linear_head(
        model_name,
        test_embeddings,
        test_labels,
        test_indices=None,
        cluster=False,
        save_results=True,
):
    # Device configuration (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load the model
    global_model = load_linear_mnist_head(device, model_name=model_name, cluster=cluster)

    # Filter test embeddings and labels
    if test_indices is not None:
        test_embeddings = test_embeddings[test_indices]
        test_labels = test_labels[test_indices]

    print('Evaluating reference model...')
    accuracy_reference, correct_ref_list, ce_loss_list = evaluate_model(
        model=global_model,
        test_embeddings=test_embeddings,
        test_labels=test_labels,
    )

    # Save the results in the results_dict
    results_dict = {
        'accuracy': accuracy_reference,
        'correct_list': correct_ref_list.tolist(),
        'ce_loss_list': ce_loss_list.tolist(),
        'model_name': model_name,
        'num_samples': len(test_labels),
    }

    if save_results:
        results_dir = get_results_dir(experiment_name='mnist_linear_head')
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
    _, test_embeddings, test_labels, _ = load_mnist_embeddings('test_data')
    _, val_embeddings, val_labels, _ = load_mnist_embeddings('validation_data')

    # Load the evaluation set
    evaluation_set = 'validation'
    evaluation_embeddings, evaluation_labels, evaluation_indices = load_evaluation_set(
        test_embeddings,
        test_labels,
        val_embeddings,
        val_labels,
        evaluation_set=evaluation_set,
        num_indices_per_class='all'
    )

    # Run the analysis
    model_name, cluster = get_name_of_mnist_linear_head()

    # Run analysis
    results = run_analysis_mnist_linear_head(
        model_name=model_name,
        test_embeddings=evaluation_embeddings,
        test_labels=evaluation_labels,
        test_indices=evaluation_indices,
        cluster=cluster,
        save_results=True,
    )

    print(f'\nReference model accuracy: {(results["accuracy"] * 100):.2f}%\n')

    # Add parameters to result
    results['evaluation_set'] = evaluation_set

    # Drop columns
    results.pop('correct_list', None)
    results.pop('ce_loss_list', None)

    # Save results in log book
    filename = os.path.splitext(os.path.basename(__file__))[0]
    log_file = f'{filename}.csv'
    log_to_csv(results, log_file)

    # Print results
    cols = ['date', 'time', 'accuracy',  'evaluation_set', 'num_samples', 'model_name']
    all_cols, df = print_tabulated(log_file, head=10, cols=cols, sort_by='accuracy')