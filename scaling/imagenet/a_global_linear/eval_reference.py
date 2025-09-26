import os
import json
import numpy as np
import torch

from scaling.models.evaluation import evaluate_model
from scaling.models.imagenet import load_linear_imagenet_model
from scaling.utils.data_loader_imagenet import load_clip_embeddings, load_evaluation_set
from scaling.utils.directory import get_results_dir
from scaling.log_book.read_and_write import log_to_csv, print_tabulated
from scaling.imagenet.parameters.param_helper import get_name_of_global_linear_model


def run_analysis_reference(
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
    global_model = load_linear_imagenet_model(device, model_name=model_name, cluster=cluster)

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
        'accuracy_reference': accuracy_reference,
        'correct_ref_list': correct_ref_list.tolist(),
        'ce_loss_list': ce_loss_list.tolist(),
        'model_name': model_name,
        'num_samples': len(test_labels),
    }

    if save_results:
        results_dir = get_results_dir(experiment_name='imagenet_reference')
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

    # Run the analysis
    model_name, cluster = get_name_of_global_linear_model()

    results = run_analysis_reference(
        model_name=model_name,
        test_embeddings=evaluation_embeddings,
        test_labels=evaluation_labels,
        test_indices=evaluation_indices,
        cluster=cluster,
        save_results=True,
    )

    print(f'\nReference model accuracy: {(results["accuracy_reference"] * 100):.2f}%\n')

    # Save results in log book
    filename = os.path.splitext(os.path.basename(__file__))[0]
    log_file = f'{filename}.csv'
    log_to_csv(results, log_file)

    # Print results
    cols = ['date', 'time', 'accuracy_reference',  'num_samples', 'model_name']
    all_cols, df = print_tabulated(log_file, head=10, cols=cols)