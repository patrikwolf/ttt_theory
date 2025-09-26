import os
import torch

from scaling.models.torch_linear_classifier import TorchLinearClassifier
from scaling.models.torch_mlp_classifier import TorchMLPClassifier


def load_linear_imagenet_model(
        device,
        model_name,
        input_dim: int=512,
        num_classes: int=1000,
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


def load_imagenet_mlp_head(
        device,
        model_name,
        input_dim: int=512,
        hidden_dim: int=1024,
        num_classes: int=1000,
        dropout_rate: float=0.5,
        cluster: bool=False,
):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if cluster:
        model_dir = os.path.join(base_dir, '../models_cluster')
    else:
        model_dir = os.path.join(base_dir, 'saved')
    torch_classifier = TorchMLPClassifier(input_dim, hidden_dim, num_classes, dropout_rate=dropout_rate)
    torch_classifier.load_state_dict(torch.load(f'{model_dir}/{model_name}.pth', map_location=device))
    return torch_classifier