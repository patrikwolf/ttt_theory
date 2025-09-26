import numpy as np


def split_dataset(labels, dataset_scales, num_classes=1000):
    # Group samples by class
    class_chunks = {k: [] for k in range(num_classes)}
    for idx, label in enumerate(labels):
        class_chunks[int(label)].append(idx)

    # Determine max samples per class available (minimum across all classes to ensure balance)
    samples_per_class = len(labels) // num_classes
    dataset_chunks = {}

    for scale in dataset_scales:
        num_samples_scaled = int(samples_per_class * scale)
        index_subset = []

        for label in class_chunks.keys():
            actual_num = min(num_samples_scaled, len(class_chunks[label]))

            # Sample the scaled number of items for this class
            index_subset.append(class_chunks[label][:actual_num])

        index_subset = np.concatenate(index_subset)

        dataset_chunks[scale] = {
            'indices': index_subset,
            'actual_scale': len(index_subset) / len(labels),
        }

    return dataset_chunks