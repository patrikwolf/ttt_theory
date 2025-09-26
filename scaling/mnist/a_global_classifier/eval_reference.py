import os
import json
import numpy as np
import torch
import torch.nn as nn

from scaling.log_book.read_and_write import log_to_csv, print_tabulated
from scaling.mnist.parameters.param_helper import get_name_of_global_mnist_model
from scaling.models.mnist import load_global_mnist_model
from scaling.utils.data_loader_mnist import load_mnist_embeddings
from scaling.utils.directory import get_results_dir


def run_analysis_global_minst(
        model_name,
        test_images,
        test_labels,
        val_images,
        val_labels,
        size_scale,
        cluster=False,
        save_results=True,
):
    # Device configuration (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load the model
    global_model = load_global_mnist_model(
        device=device,
        model_name=model_name,
        size_scale=size_scale,
        cluster=cluster,
        only_head=False,
    )

    print('Evaluating reference model on the test set...')
    results_test = evaluate_mnist_model(
        model=global_model,
        test_embeddings=test_images,
        test_labels=test_labels,
    )

    print('Evaluating reference model on the test set...')
    results_val = evaluate_mnist_model(
        model=global_model,
        test_embeddings=val_images,
        test_labels=val_labels,
    )

    # Save the results in the results_dict
    results_dict = {
        'accuracy_test': results_test['accuracy'],
        'correct_list_test': results_test['correct_list'],
        'ce_loss_list_test': results_test['ce_loss_list'],
        'num_samples_test': len(test_labels),
        'accuracy_val': results_val['accuracy'],
        'correct_list_val': results_val['correct_list'],
        'ce_loss_list_val': results_val['ce_loss_list'],
        'num_samples_val': len(val_labels),
        'model_name': model_name,
        'size_scale': size_scale,
    }

    if save_results:
        results_dir = get_results_dir(experiment_name='mnist_reference')
        results_file = f'{results_dir}/results.json'
        with open(results_file, 'w') as f:
            json.dump(results_dict, f)
        print(f'Results saved to {results_file}')
    else:
        print(f'---> Result saving within evaluation function is disabled.')

    return results_dict


def evaluate_mnist_model(
        model,
        test_embeddings,
        test_labels
):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')
    with torch.no_grad():
        # Forward pass
        outputs = model(test_embeddings)

        # CE loss
        ce_loss_list = criterion(outputs, test_labels)

        # Get the predicted class
        _, predicted = torch.max(outputs, 1)

        # Calculate accuracy
        correct_ref_list = (predicted == test_labels).int()
        accuracy = correct_ref_list.float().mean()

    # Add results to dictionary
    results = {
        'accuracy': float(accuracy),
        'correct_list': correct_ref_list.tolist(),
        'ce_loss_list': ce_loss_list.tolist(),
    }

    return results


if __name__ == '__main__':
    # Fix random seed for reproducibility
    np.random.seed(42)

    # Load the embeddings and labels
    print(f'Loading images and labels...')
    test_images, _, test_labels, _ = load_mnist_embeddings('test_data')
    val_images, _, val_labels, _ = load_mnist_embeddings('validation_data')

    # Get the global model
    model_name, size_scale, cluster = get_name_of_global_mnist_model()

    # Run the analysis
    results = run_analysis_global_minst(
        model_name=model_name,
        size_scale=size_scale,
        test_images=test_images,
        test_labels=test_labels,
        val_images=val_images,
        val_labels=val_labels,
        cluster=cluster,
        save_results=True,
    )

    print('\n' + '*' * 80)
    print(f'Test accuracy of reference model: {(results["accuracy_test"] * 100):.2f}%')
    print(f'Validation accuracy of reference model: {(results["accuracy_val"] * 100):.2f}%')
    print('*' * 80 + '\n')

    # Drop some keys
    drop_keys = ['correct_list_test', 'ce_loss_list_test', 'correct_list_val', 'ce_loss_list_val']
    for key in drop_keys:
        if key in results:
            results.pop(key)

    # Save results in log book
    filename = os.path.splitext(os.path.basename(__file__))[0]
    log_file = f'mnist_{filename}.csv'
    log_to_csv(results, log_file, mnist=True)

    # Print results
    cols = ['date', 'time', 'accuracy_test', 'accuracy_val', 'num_samples_test', 'num_samples_val', 'size_scale', 'model_name']
    all_cols, df = print_tabulated(log_file, head=10, cols=cols)