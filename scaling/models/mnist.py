import os
import re
import torch

from scaling.models.cnn import FlexibleLeNet
from scaling.models.torch_linear_classifier import TorchLinearClassifier


def load_raw_mnist_model(model_type, num_classes):
    # Regex pattern for FlexibleLeNet_X.XX (X.XX must have exactly two decimal places)
    flexible_pattern = r"^FlexibleLeNet_\d+\.\d{2}$"

    if re.match(flexible_pattern, model_type):
        return FlexibleLeNet(num_classes, size_scale=float(model_type.split("_")[1]))
    else:
        raise ValueError(f"Invalid model_type format: '{model_type}'. Expected format like 'FlexibleLeNet_X.XX'.")


def load_global_mnist_model(
        device,
        model_name,
        size_scale,
        num_classes: int=10,
        cluster=False,
        only_head=True,
):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if cluster:
        model_dir = os.path.join(base_dir, '../models_cluster')
    else:
        model_dir = os.path.join(base_dir, 'saved')
    torch_classifier = FlexibleLeNet(num_classes, size_scale)
    torch_classifier.load_state_dict(torch.load(f'{model_dir}/{model_name}.pth', map_location=device))

    if only_head:
        torch_classifier = torch_classifier.classifier

    return torch_classifier


def load_linear_mnist_head(
        device,
        model_name,
        input_dim: int=50,
        num_classes: int=10,
        cluster=False,
):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if cluster:
        model_dir = os.path.join(base_dir, '../models_cluster')
    else:
        model_dir = os.path.join(base_dir, 'saved')
    torch_classifier = TorchLinearClassifier(input_dim, num_classes)
    torch_classifier.load_state_dict(torch.load(f'{model_dir}/{model_name}.pth', map_location=device))
    return torch_classifier