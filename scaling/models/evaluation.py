import torch
import torch.nn as nn


def evaluate_model(
        model,
        test_embeddings,
        test_labels
):
    # Set to evaluation mode
    model.eval()

    # Define loss function
    criterion = nn.CrossEntropyLoss(reduction='none')

    # Disable gradient calculation
    with torch.no_grad():
        # Forward pass
        outputs = model(test_embeddings)

        # CE loss
        ce_loss_list = criterion(outputs, test_labels)

        # Get the predicted class
        _, predicted = torch.max(outputs, 1)

        # Calculate accuracy
        correct_list = (predicted == test_labels).int()
        accuracy = correct_list.float().mean()

    return float(accuracy), correct_list, ce_loss_list