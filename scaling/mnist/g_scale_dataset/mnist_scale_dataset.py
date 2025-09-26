import os
import time
import torch
import json
import numpy as np

from pathlib import Path
from datetime import datetime
from scaling.log_book.read_and_write import log_to_csv, print_tabulated
from scaling.mnist.b_linear_head.train_mnist_linear_head import train_mnist_linear_head
from scaling.mnist.c_ttt_linear_head.ttt_linear import run_analysis_lft_erm
from scaling.models.evaluation import evaluate_model
from scaling.utils.data_loader import load_evaluation_set
from scaling.utils.data_loader_mnist import load_mnist_embeddings
from scaling.utils.directory import get_results_dir, get_models_dir
from scaling.utils.scale_dataset import split_dataset


def run_analysis_dataset_scaling(
        models,
        train_embeddings,
        train_labels,
        evaluation_embeddings,
        evaluation_labels,
        evaluation_indices,
        evaluation_set,
        dataset_scales,
        num_classes,
        save_results=True,
):
    # Device configuration (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Get results directory
    results_dir = get_results_dir('mnist_scale_dataset', create_dir=False).parent
    results_dirs = {}

    # Shuffle the dataset
    indices = torch.randperm(len(train_labels))
    train_embeddings = train_embeddings[indices]
    train_labels = train_labels[indices]

    # Split dataset into different scales
    print('Splitting dataset into different scales...')
    dataset_chunks = split_dataset(train_labels, dataset_scales, num_classes=num_classes)

    # Iterate over the classifiers
    for model in models:
        start = time.time()

        # Prepare the results dictionary
        results_dict = {
            'model': model['name'],
            'model_paths': [],
            'intended_scales': [],
            'actual_scales': [],
            'num_samples': [],
            'accuracies': [],
            'correct_list': [],
        }

        # Model name
        model_name = model['name'].lower().replace(' ', '_')

        # Iterate over different dataset scales
        for intended_scale, chunk in dataset_chunks.items():
            print('*' * 80)
            print(f'Processing dataset scale: {chunk["actual_scale"]:.4f} (intended: {intended_scale})')
            print('*' * 80)
            start_loop = time.time()

            # Get the embeddings and labels for the current scale
            train_emb_chunk = train_embeddings[chunk['indices']]
            train_label_chunk = train_labels[chunk['indices']]

            # Train the classifier on the current subset
            print(f'Training: {model["name"]} on scale {chunk["actual_scale"]:.4f} (intended: {intended_scale})')
            accuracy, correct_list, model_path, params = model['eval_fct'](
                train_embeddings=train_emb_chunk,
                train_labels=train_label_chunk,
                evaluation_embeddings=evaluation_embeddings,
                evaluation_labels=evaluation_labels,
                evaluation_indices=evaluation_indices,
                device=device,
            )

            # Append the results for this model
            results_dict['model_paths'].append(model_path)
            results_dict['intended_scales'].append(intended_scale)
            results_dict['actual_scales'].append(chunk['actual_scale'])
            results_dict['num_samples'].append(len(train_label_chunk))
            results_dict['accuracies'].append(accuracy)
            results_dict['correct_list'].append(correct_list.tolist())

            # Measure time
            end_loop = time.time()
            print(f'The inner loop took {end_loop - start_loop:.2f} seconds')
            print('*' * 80)

        # Add learning parameters to the results
        results_dict['params'] = params
        results_dict['evaluation_set'] = evaluation_set

        if save_results:
            # Save the results in the results array
            output_dir = Path(f'{results_dir}/{model_name}/{datetime.now().strftime("%Y-%m-%d__%H-%M-%S")}')
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = f'{output_dir}/results.json'
            with open(output_file, 'w') as f:
                json.dump(results_dict, f)
            print(f'Results for model {model["name"]} saved to {output_file}')

            results_dirs[model_name] = {
                'output': output_file,
            }
        else:
            print(f'---> Result saving within evaluation function is disabled.')

        # Measure time
        end = time.time()
        print('*' * 80)
        print(f'The loop for the entire model took {end - start:.2f} seconds')
        print('*' * 80)

    return results_dict, results_dirs


def train_and_eval_linear_classifier(train_embeddings, train_labels, evaluation_embeddings, evaluation_labels,
                                     evaluation_indices, device):
    # Hyperparameters
    learning_rate = 0.005
    batch_size = 400
    num_epochs = 100

    # Train the PyTorch linear classifier
    torch_classifier = train_mnist_linear_head(
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        val_embeddings=evaluation_embeddings,
        val_labels=evaluation_labels,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        use_wandb=False,
    )

    # Save the PyTorch model
    model_dir = get_models_dir()
    model_dir = Path(f'{model_dir}/scaled_dataset')
    model_dir.mkdir(parents=True, exist_ok=True)
    model_name = f'mnist_linear_classifier_ns{len(train_labels)}.pth'
    torch.save(torch_classifier.state_dict(), f'{model_dir}/{model_name}')

    # Store the parameters used for training
    params = {
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
    }

    # Evaluate the classifier on the evaluation set
    evaluation_embeddings = evaluation_embeddings[evaluation_indices].to(device)
    evaluation_labels = evaluation_labels[evaluation_indices].to(device)
    accuracy, correct_list, _ = evaluate_model(torch_classifier, evaluation_embeddings, evaluation_labels)

    return accuracy, correct_list, model_name, params


def train_and_eval_lft_linear_head(train_embeddings, train_labels, evaluation_embeddings, evaluation_labels,
                                   evaluation_indices, device):
    # Parameters for local fine-tuning
    num_neighbors = 80
    optimization = {
        'erm': {
            'type': 'erm',
            'learning_rate': 2e-2,
            'epochs': 500,
        }
    }

    # Use the pre-trained model
    model_name = f'scaled_dataset/mnist_linear_classifier_ns{len(train_labels)}'

    # Run the local fine-tuning analysis
    results = run_analysis_lft_erm(
        model_name=model_name,
        size_scale=None,
        linear_head=True,
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        test_embeddings=evaluation_embeddings,
        test_labels=evaluation_labels,
        test_indices=evaluation_indices,
        num_neighbors=num_neighbors,
        optimization=optimization,
        wandb_indices=[],
        cluster=False,
        save_results=False,
    )

    # Extract accuracy and correct list from results
    accuracy = results['erm']['accuracy']
    correct_list = results['erm']['correct_list']

    # Parameters
    params = {
        'num_neighbors': num_neighbors,
        'optimization': optimization,
        'model_name': model_name,
    }

    return accuracy, np.array(correct_list), model_name, params


if __name__ == '__main__':
    # Fix random seed for reproducibility
    np.random.seed(42)

    # Number of classes
    num_classes = 10

    # Load the embeddings and labels
    print(f'Loading embeddings and labels...')
    _, train_embeddings, train_labels, _ = load_mnist_embeddings('training_data')
    _, test_embeddings, test_labels, _ = load_mnist_embeddings('test_data')
    _, val_embeddings, val_labels, _ = load_mnist_embeddings('validation_data')

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

    # Dataset scales to evaluate
    dataset_scales = [0.01, 0.03, 0.1, 0.3, 1.0]

    # Selection of models to evaluate
    models = [
        {
            'name': 'Global linear classifier',
            'eval_fct': train_and_eval_linear_classifier,
        },
        {
            'name': 'LFT linear head',
            'eval_fct': train_and_eval_lft_linear_head,
        },
    ]

    # Run the analysis
    results, result_dirs = run_analysis_dataset_scaling(
        models=models,
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        evaluation_embeddings=evaluation_embeddings,
        evaluation_labels=evaluation_labels,
        evaluation_indices=evaluation_indices,
        evaluation_set=evaluation_set,
        dataset_scales=dataset_scales,
        num_classes=num_classes,
        save_results=True,
    )

    # Add parameters to results
    result_dirs['evaluation_set'] = evaluation_set
    result_dirs['num_samples'] = len(evaluation_indices)

    # Print the results
    print(json.dumps(result_dirs, indent=4))

    # Save results in log book
    filename = os.path.splitext(os.path.basename(__file__))[0]
    log_file = f'{filename}.csv'
    log_to_csv(result_dirs, log_file)

    # Print results
    cols = []
    all_cols, df = print_tabulated(log_file, head=10)