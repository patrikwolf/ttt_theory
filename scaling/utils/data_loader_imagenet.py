import os
import torch

from scaling.utils.data_loader import load_evaluation_set


def load_clip_embeddings(file_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    torch_data = torch.load(f'{base_dir}/../data/imagenet/{file_name}.pt')
    return torch_data['embeddings'], torch_data['labels']


if __name__ == '__main__':
    # Load the embeddings and labels
    print(f'Loading embeddings and labels...')
    train_embeddings, train_labels = load_clip_embeddings('training_data')
    test_embeddings, test_labels = load_clip_embeddings('test_data')
    val_embeddings, val_labels = load_clip_embeddings('validation_data')

    num_indices_per_class = 2

    # Load the evaluation set
    evaluation_embeddings, evaluation_labels, evaluation_indices = load_evaluation_set(
        test_embeddings,
        test_labels,
        val_embeddings,
        val_labels,
        evaluation_set='test',
        num_indices_per_class=num_indices_per_class
    )

    print(f'--> Number of indices per class: {num_indices_per_class}')
    print(f'--> Size of evaluation set: {len(evaluation_embeddings)}')