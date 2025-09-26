import torch

from scaling.data.imagenet.data_generator import load_batched_clip_embeddings


def test_val_split():
    # Load data
    train_embeddings, train_labels = load_batched_clip_embeddings('clip_train_chunks')
    val_embeddings, val_labels = load_batched_clip_embeddings('clip_val_chunks')

    # Shuffle the dataset
    num_samples = train_embeddings.shape[0]
    perm = torch.randperm(num_samples)
    shuffled_embeddings = train_embeddings[perm]
    shuffled_labels = train_labels[perm]

    # Get unique classes
    unique_classes = torch.unique(shuffled_labels)
    n_classes = unique_classes.numel()
    samples_per_class = 50000 // n_classes

    # Select 50 indices per class for the test set
    class_counts = {int(cls.item()): 0 for cls in unique_classes}
    selected_indices = []

    for i, label in enumerate(shuffled_labels):
        label_int = int(label.item())
        if class_counts[label_int] < samples_per_class:
            selected_indices.append(i)
            class_counts[label_int] += 1
        if sum(class_counts.values()) >= 50000:
            break

    # Convert to tensor
    selected_indices = torch.tensor(selected_indices)

    # Get test set
    test_embeddings = shuffled_embeddings[selected_indices]
    test_labels = shuffled_labels[selected_indices]

    # Get remaining training set
    mask = torch.ones(num_samples, dtype=torch.bool)
    mask[selected_indices] = False

    remaining_embeddings = shuffled_embeddings[mask]
    remaining_labels = shuffled_labels[mask]

    # Save datasets
    torch.save({"embeddings": remaining_embeddings, "labels": remaining_labels},'training_data.pt')
    torch.save({"embeddings": test_embeddings, "labels": test_labels}, 'test_data.pt')
    torch.save({"embeddings": val_embeddings, "labels": val_labels}, 'validation_data.pt')


if __name__ == "__main__":
    # Create test and validation splits
    test_val_split()
    print('Done!')