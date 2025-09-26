import os
import time
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import faiss
import wandb
import argparse

from datetime import datetime
from scaling.models.imagenet import load_linear_imagenet_model
from scaling.imagenet.parameters.param_helper import get_name_of_global_linear_model
from scaling.log_book.read_and_write import log_to_csv, print_tabulated
from scaling.models.evaluation import evaluate_model
from scaling.utils.data_loader import load_evaluation_set
from scaling.utils.data_loader_imagenet import load_clip_embeddings
from scaling.utils.directory import get_results_dir


def run_analysis_ttt(
        model_name,
        train_embeddings,
        train_labels,
        test_embeddings,
        test_labels,
        test_indices,
        num_neighbors,
        optimization,
        wandb_indices,
        save_results=True,
):
    # Device configuration (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load the model
    global_model = load_linear_imagenet_model(device, model_name=model_name)
    global_model.to(device)

    # Build FAISS index
    faiss_index = faiss.IndexFlatL2(train_embeddings.shape[1])
    faiss_index.add(train_embeddings)

    print('Evaluating locally fine-tuned model...')
    results_dict = evaluate_model_with_ttt(
        torch_classifier=global_model,
        device=device,
        faiss_index=faiss_index,
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        test_embeddings=test_embeddings,
        test_labels=test_labels,
        test_indices=test_indices,
        num_neighbors=num_neighbors,
        optimization=optimization,
        wandb_indices=wandb_indices,
    )

    # Add parameters to results dictionary
    results_dict['model_name'] = model_name
    results_dict['num_samples'] = len(test_indices)
    results_dict['num_neighbors'] = num_neighbors
    results_dict['optimization'] = optimization

    # Save the results to a JSON file
    if save_results:
        results_dir = get_results_dir(experiment_name='imagenet_global_ttt')
        results_file = f'{results_dir}/results.json'
        with open(results_file, 'w') as f:
            json.dump(results_dict, f)
        print(f'Results saved to {results_file}')
    else:
        print(f'---> Result saving within evaluation function is disabled.')

    return results_dict


def evaluate_model_with_ttt(
        torch_classifier,
        device,
        faiss_index,
        train_embeddings,
        train_labels,
        test_embeddings,
        test_labels,
        test_indices,
        num_neighbors,
        optimization,
        wandb_indices,
):
    # Initialize counters for correct predictions
    num_samples = len(test_indices)
    stats = {
        'correct_list': np.zeros(num_samples, dtype=int),
        'ce_loss_list': np.zeros(num_samples),
    }
    results_dict = {key: copy.deepcopy(stats) for key in optimization.keys()}
    results_dict['global'] = {
        'correct_list': np.zeros(num_samples, dtype=int),
        'ce_loss_list': np.zeros(num_samples),
    }

    start = time.time()
    for idx, test_idx in enumerate(test_indices):
        if (idx + 1) % 100 == 0:
            print(f'Evaluating sample {idx + 1}/{num_samples} ({100 * (idx + 1) / num_samples:.2f}%) '
                  f'after {((time.time() - start) / 60):.2f} minutes')

        # Get the test sample embedding and its true label
        test_sample_embedding = test_embeddings[test_idx].reshape(1, -1)
        true_label = test_labels[test_idx]
        test_label = true_label.to(device).unsqueeze(0)

        # Make prediction with the global model
        test_tensor = test_sample_embedding.to(device)
        predicted_label, _, ce_loss = evaluate_model(
            model=torch_classifier,
            test_embeddings=test_tensor,
            test_labels=test_label,
        )

        results_dict['global']['correct_list'][idx] = int(predicted_label == true_label)
        results_dict['global']['ce_loss_list'][idx] = ce_loss.item()

        if (idx + 1) % 100 == 0:
            print(f'Current accuracy of global linear classifier: {(100 * np.sum(results_dict["global"]["correct_list"]) / (idx + 1)):.2f}')

        # Get prediction from the local fine-tuning method
        results = locally_fine_tune_and_predict_torch(
            test_sample_embedding,
            true_label,
            torch_classifier,
            train_embeddings,
            train_labels,
            faiss_index,
            device=device,
            num_neighbors=num_neighbors,
            optimization=optimization,
            use_wandb=(idx in wandb_indices),
        )

        for model, params in results.items():
            fine_tuned_pred = params['predicted_label']
            ce_loss = params['ce_loss']

            # Update the results dictionary
            results_dict[model]['correct_list'][idx] = int(fine_tuned_pred == true_label)
            results_dict[model]['ce_loss_list'][idx] = ce_loss

            if (idx + 1) % 100 == 0:
                print(f'Current accuracy of locally fine-tuned linear classifier with "{params["type"]}" optimization: {(100 * np.sum(results_dict[model]["correct_list"]) / (idx + 1)):.2f}')

        if (idx + 1) % 100 == 0:
            print('*' * 100)

    # Calculate the accuracy for both global and local fine-tuning
    results_dict['global']['accuracy'] = float(np.sum(results_dict['global']['correct_list']) / num_samples)
    results_dict['global']['correct_list'] = results_dict['global']['correct_list'].tolist()
    results_dict['global']['ce_loss_list'] = results_dict['global']['ce_loss_list'].tolist()

    for model, params in optimization.items():
        results_dict[model]['accuracy'] = float(np.sum(results_dict[model]['correct_list']) / num_samples)
        results_dict[model]['correct_list'] = results_dict[model]['correct_list'].tolist()
        results_dict[model]['ce_loss_list'] = results_dict[model]['ce_loss_list'].tolist()

    return results_dict


def locally_fine_tune_and_predict_torch(
        test_sample,
        test_label,
        global_model,
        train_embeddings,
        train_labels,
        faiss_index,
        device,
        num_neighbors,
        optimization,
        use_wandb,
):
    # Fetch the k nearest neighbors
    _, indices = faiss_index.search(test_sample, num_neighbors)
    neighbor_indices = indices[0]
    neighbor_embeddings = train_embeddings[neighbor_indices]
    neighbor_labels = train_labels[neighbor_indices]

    # Sort neighbor by increasing distance to the test sample
    distances = torch.norm(neighbor_embeddings - test_sample, dim=1)
    sorted_indices = torch.argsort(distances)
    neighbor_embeddings = neighbor_embeddings[sorted_indices]
    neighbor_labels = neighbor_labels[sorted_indices]

    # Move the data to the appropriate device
    test_tensor = test_sample.to(device)
    test_label = test_label.to(device).unsqueeze(0)

    # Iterate over optimization methods
    results = {}
    for opt_type, opt_params in optimization.items():
        # Initialize the results dictionary for this optimization type
        results[opt_type] = {
            'type': opt_type,
        }

        # Local fine-tuning
        local_model = locally_fine_tune_torch_model(
            global_model=global_model,
            neighbor_embeddings=neighbor_embeddings,
            neighbor_labels=neighbor_labels,
            device=device,
            optimization_params=opt_params,
            use_wandb=use_wandb,
        )

        # Make the final prediction using the fine-tuned local model
        predicted_label, _, ce_loss = evaluate_model(
            model=local_model,
            test_embeddings=test_tensor,
            test_labels=test_label,
        )

        # Add results to dictionary
        results[opt_type]['predicted_label'] = predicted_label
        results[opt_type]['ce_loss'] = ce_loss.item()

    return results


def locally_fine_tune_torch_model(
        global_model,
        neighbor_embeddings,
        neighbor_labels,
        device,
        optimization_params,
        use_wandb,
):
    # Create a deep copy of the global model to fine-tune.
    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()

    # Set up a new optimizer and loss function for the local model
    optimizer = torch.optim.Adam(local_model.parameters(), lr=optimization_params['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    # Move neighbor tensors to the appropriate device
    neighbor_tensors = neighbor_embeddings.to(device)
    neighbor_labels_tensors = neighbor_labels.long().to(device)

    # Run the fine-tuning loop for a few epochs
    if optimization_params['type'] == 'erm':
        for epoch in range(optimization_params['epochs']):
            # Forward pass
            outputs = local_model(neighbor_tensors)
            loss = criterion(outputs, neighbor_labels_tensors)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'loss': loss.item(),
                })
    elif optimization_params['type'] == 'sequential':
        for i in range(len(neighbor_tensors)):
            input_i = neighbor_tensors[i].unsqueeze(0)
            label_i = neighbor_labels_tensors[i].unsqueeze(0)

            # Forward pass
            output_i = local_model(input_i)
            loss = criterion(output_i, label_i)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    else:
        raise ValueError(f'Unknown optimization method: {optimization_params["type"]}')

    return local_model


if __name__ == '__main__':
    # Fix random seed for reproducibility
    np.random.seed(42)

    # Datetime
    now = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_id", type=int, required=False, default=0)
    parser.add_argument("--num_shards", type=int, required=False, default=1)
    parser.add_argument("--datetime", type=str, required=False, default=now)
    args = parser.parse_args()

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

    # Split indices into shards
    shards = np.array_split(evaluation_indices, args.num_shards)
    eval_shards = shards[args.shard_id]
    save_results = (args.num_shards == 1)

    # Parameters for local fine-tuning
    num_neighbors = 60

    # Optimization parameters
    optimization = {
        'erm': {
            'type': 'erm',
            'learning_rate': 1e-2,
            'epochs': 50,
        },
        'sequential': {
            'type': 'sequential',
            'learning_rate': 3 * 1e-3,
        },
    }

    # Model name
    model_name, cluster = get_name_of_global_linear_model()

    # Initialize Weights & Biases
    wandb.init(project='ttt-linear-head', config={
        'learning_rate': optimization['erm']['learning_rate'],
        'num_epochs': optimization['erm']['epochs'],
        'num_neighbors': num_neighbors,
        'model_name': model_name
    })

    print(f'Run TTT analysis...')
    start = time.time()
    results = run_analysis_ttt(
        model_name=model_name,
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        test_embeddings=evaluation_embeddings,
        test_labels=evaluation_labels,
        test_indices=eval_shards,
        num_neighbors=num_neighbors,
        optimization=optimization,
        wandb_indices=[0, 1, 2],
        save_results=save_results,
    )
    end = time.time()
    print(f'Finished in {end - start} seconds')

    # Finish Weights & Biases run
    wandb.finish()

    print('\n' + '*' * 80)
    print(f'Accuracy of global model on {results["num_samples"]} samples: {(results["global"]["accuracy"] * 100):.2f}%')
    for opt_type, opt_params in optimization.items():
        print(f'Accuracy of locally fine-tuned linear classifier with "{opt_params["type"]}" optimization on {results["num_samples"]} samples: {(results[opt_type]["accuracy"] * 100):.2f}%')
    print('*' * 80 + '\n')

    # Add parameters to results
    results['evaluation_set'] = evaluation_set

    # Save results
    results_dir = get_results_dir(experiment_name='imagenet_global_ttt', timestamp=args.datetime)
    results_file = f'{results_dir}/shard_{args.shard_id}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f)
    print(f'Results saved to {results_file}')
    print(f'Results generated from file "{os.path.basename(__file__)}" with shard ID {args.shard_id} and '
          f'datetime {args.datetime}')

    if save_results:
        # Prepare results for logging
        results['accuracy_global'] = results['global']['accuracy']
        results['accuracy_lft_erm'] = results['erm']['accuracy']
        results['accuracy_lft_sequential'] = results['sequential']['accuracy']

        # Save results in log book
        filename = os.path.splitext(os.path.basename(__file__))[0]
        log_file = f'{filename}.csv'
        log_to_csv(results, log_file)

        # Print results
        cols = ['date', 'time', 'model_name', 'num_samples', 'num_neighbors', 'accuracy_global', 'accuracy_lft_erm', 'accuracy_lft_sequential']
        all_cols, df = print_tabulated(log_file, cols=cols, head=10)
    else:
        print(f'---> Result not logged independently since we have multiple shards.')