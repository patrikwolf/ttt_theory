import time
import torch
import json
import numpy as np

from pathlib import Path
from datetime import datetime
from scaling.imagenet.a_global_linear.train_global_linear_model import train_torch_linear_classifier
from scaling.imagenet.b_ttt_linear_head.ttt_linear_head import run_analysis_ttt
from scaling.imagenet.parameters.param_helper import get_optimal_mlp_parameters
from scaling.imagenet.c_mlp_heads.train_global_mlp_head import train_torch_mlp_classifier
from scaling.imagenet.d_ttt_mlp_heads.ttt_mlp_head import run_analysis_ttt_mlp_heads
from scaling.models.evaluation import evaluate_model
from scaling.utils.data_loader import load_evaluation_set
from scaling.utils.data_loader_imagenet import load_clip_embeddings
from scaling.utils.directory import get_results_dir, get_models_dir
from scaling.utils.scale_dataset import split_dataset


def run_analysis_dataset_scaling(
        models,
        train_embeddings,
        train_labels,
        evaluation_embeddings,
        evaluation_labels,
        evaluation_set,
        dataset_scales,
):
    # Device configuration (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Get results directory
    results_dir = get_results_dir('imagenet_scale_dataset').parent
    results_dirs = {}

    # Shuffle the dataset
    indices = torch.randperm(len(train_labels))
    train_embeddings = train_embeddings[indices]
    train_labels = train_labels[indices]

    # Split dataset into different scales
    print('Splitting dataset into different scales...')
    dataset_chunks = split_dataset(train_labels, dataset_scales)

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

        if 'hidden_dim' in model:
            model_name = model['name'].lower().replace(' ', '_') + f'_hidden_dim_{model["hidden_dim"]}'
            results_dict['hidden_dim'] = model['hidden_dim']
        else:
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
            if 'hidden_dim' in model:
                print(f'Training: {model["name"]} with hidden dimension {model["hidden_dim"]} on scale {chunk["actual_scale"]:.4f} (intended: {intended_scale})')
                accuracy, correct_list, model_path, params = model['eval_fct'](train_emb_chunk, train_label_chunk, evaluation_embeddings, evaluation_labels, device, hidden_dim=model['hidden_dim'])
            else:
                print(f'Training: {model["name"]} on scale {chunk["actual_scale"]:.4f} (intended: {intended_scale})')
                accuracy, correct_list, model_path, params = model['eval_fct'](train_emb_chunk, train_label_chunk, evaluation_embeddings, evaluation_labels, device)

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

        # Measure time
        end = time.time()
        print('*' * 80)
        print(f'The loop for the entire model took {end - start:.2f} seconds')
        print('*' * 80)

    return results_dirs


def train_and_eval_linear_classifier(train_embeddings, train_labels, evaluation_embeddings, evaluation_labels, device):
    # Hyperparameters
    learning_rate = 1e-3
    batch_size = 250
    num_epochs = 50

    # Train the PyTorch linear classifier
    torch_classifier = train_torch_linear_classifier(
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
    model_name = f'torch_linear_classifier_imagenet_ns{len(train_labels)}.pth'
    torch.save(torch_classifier.state_dict(), f'{model_dir}/{model_name}')

    # Store the parameters used for training
    params = {
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
    }

    # Evaluate the classifier on the evaluation set
    evaluation_embeddings = evaluation_embeddings.to(device)
    evaluation_labels = evaluation_labels.to(device)
    accuracy, correct_list, _ = evaluate_model(torch_classifier, evaluation_embeddings, evaluation_labels)

    return accuracy, correct_list, model_name, params


def train_and_eval_mlp_classifier(train_embeddings, train_labels, evaluation_embeddings, evaluation_labels, device, hidden_dim):
    # Parameters for the analysis
    model_param_list = get_optimal_mlp_parameters()

    # Get optimal hyperparameters for the given hidden dimension
    if str(hidden_dim) not in model_param_list:
        raise ValueError(f'Hidden dimension {hidden_dim} not found in optimal parameters list.')
    hyperparameters = model_param_list[str(hidden_dim)]

    # Train the PyTorch MLP classifier
    torch_mlp_classifier = train_torch_mlp_classifier(
        train_tensors=train_embeddings,
        train_labels=train_labels,
        val_tensors=evaluation_embeddings,
        val_labels=evaluation_labels,
        hidden_dim=hidden_dim,
        learning_rate=hyperparameters['learning_rate'],
        weight_decay=hyperparameters['weight_decay'],
        batch_size=hyperparameters['batch_size'],
        num_epochs=hyperparameters['num_epochs'],
        dropout_rate=hyperparameters['dropout_rate'],
        num_classes=1000,
        use_wandb=False,
    )

    # Save the PyTorch model
    model_name = f'torch_mlp_classifier_hd{hidden_dim}_ns{len(train_labels)}'
    model_dir = get_models_dir()
    model_dir = Path(f'{model_dir}/scaled_dataset')
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch_mlp_classifier.state_dict(), f'{model_dir}/{model_name}.pth')

    # Store the parameters used for training
    hyperparameters['hidden_dim'] = hidden_dim

    # Evaluate the classifier on the evaluation set
    evaluation_embeddings = evaluation_embeddings.to(device)
    evaluation_labels = evaluation_labels.to(device)
    accuracy, correct_list, _ = evaluate_model(torch_mlp_classifier, evaluation_embeddings, evaluation_labels)

    return accuracy, correct_list, model_name, hyperparameters


def train_and_eval_ttt_linear_head(train_embeddings, train_labels, evaluation_embeddings, evaluation_labels, device):
    # Parameters for local fine-tuning
    num_neighbors = 600
    optimization = {
        'erm': {
            'type': 'erm',
            'learning_rate': 2e-2,
            'epochs': 50,
        }
    }

    # Use the pre-trained model
    model_name = f'scaled_dataset/torch_linear_classifier_imagenet_ns{len(train_labels)}'

    # Run the local fine-tuning analysis
    results = run_analysis_ttt(
        model_name=model_name,
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        test_embeddings=evaluation_embeddings,
        test_labels=evaluation_labels,
        test_indices=evaluation_indices,
        num_neighbors=num_neighbors,
        optimization=optimization,
        wandb_indices=[],
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


def train_and_eval_ttt_mlp_head(train_embeddings, train_labels, evaluation_embeddings, evaluation_labels, device, hidden_dim):
    # Parameters for local fine-tuning
    num_neighbors = 100
    optimization = {
        'num_neighbors': num_neighbors,
        'finetune_epochs': 50,
        'finetune_lr': 5e-3,
    }

    # Use the pre-trained model
    model_name = f'scaled_dataset/torch_mlp_classifier_hd{hidden_dim}_ns{len(train_labels)}'

    # Run the local fine-tuning analysis
    results = run_analysis_ttt_mlp_heads(
        model_name=model_name,
        hidden_dim=hidden_dim,
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        test_embeddings=evaluation_embeddings,
        test_labels=evaluation_labels,
        test_indices=evaluation_indices,
        num_neighbors=num_neighbors,
        optimization=optimization,
        model_on_cluster=False,
        use_wandb=False,
        save_results=False,
    )

    # Extract accuracy and correct list from results
    accuracy = results['lft']['accuracy']
    correct_list = results['lft']['correct_list']

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

    # Dataset scales to evaluate
    dataset_scales = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]

    # Selection of models to evaluate
    models = [
        {
            'name': 'Linear head',
            'eval_fct': train_and_eval_linear_classifier,
        },
        {
            'name': 'TTT linear head',
            'eval_fct': train_and_eval_ttt_linear_head,
        },
        {
            'name': 'MLP head',
            'eval_fct': train_and_eval_mlp_classifier,
            'hidden_dim': 250,
        },
        {
            'name': 'TTT MLP head',
            'eval_fct': train_and_eval_ttt_mlp_head,
            'hidden_dim': 250,
        },
    ]

    # Run the analysis
    results = run_analysis_dataset_scaling(
        models=models,
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        evaluation_embeddings=evaluation_embeddings,
        evaluation_labels=evaluation_labels,
        evaluation_set=evaluation_set,
        dataset_scales=dataset_scales,
    )

    # Add parameters to results
    results['evaluation_set'] = evaluation_set
    results['num_samples'] = len(evaluation_indices)

    # Print the results
    print(json.dumps(results, indent=4))