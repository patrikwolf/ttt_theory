import numpy as np


def load_evaluation_set(
        test_embeddings,
        test_labels,
        val_embeddings,
        val_labels,
        evaluation_set,
        num_indices_per_class,
        num_classes=1000,
):
    if evaluation_set == 'test':
        evaluation_embeddings = test_embeddings
        evaluation_labels = test_labels
    elif evaluation_set == 'validation':
        evaluation_embeddings = val_embeddings
        evaluation_labels = val_labels
    else:
        raise ValueError(f'Unknown evaluation set: {evaluation_set}')

    if num_indices_per_class == 'all':
        evaluation_indices = np.arange(len(evaluation_labels))
    elif isinstance(num_indices_per_class, int):
        evaluation_indices = subsample_indices(evaluation_labels, num_samples_per_class=num_indices_per_class, num_classes=num_classes)
    else:
        raise ValueError(f'Unknown number of indices per class: {num_indices_per_class}')

    return evaluation_embeddings, evaluation_labels, evaluation_indices


def subsample_indices(test_labels, num_samples_per_class=2, num_classes=1000):
    indices = []
    class_counts = {k: 0 for k in range(num_classes)}

    for idx, label in enumerate(test_labels):
        label = int(label)
        if label < 0:
            print(label)
        if class_counts[label] < num_samples_per_class:
            class_counts[label] += 1
            indices.append(idx)

    return np.array(indices)


if __name__ == '__main__':
    test_embeddings = np.random.rand(100, 64)
    test_labels = np.random.randint(0, 10, size=100)
    val_embeddings = np.random.rand(50, 64)
    val_labels = np.random.randint(0, 10, size=50)
    evaluation_set = 'validation'
    num_indices_per_class = 1

    # Load the evaluation set
    evaluation_embeddings, evaluation_labels, evaluation_indices = load_evaluation_set(
        test_embeddings,
        test_labels,
        val_embeddings,
        val_labels,
        evaluation_set,
        num_indices_per_class,
        num_classes=10,
    )

    print(evaluation_indices)