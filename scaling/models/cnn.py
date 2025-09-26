import torch
import torch.nn as nn

from abc import ABC, abstractmethod


class ExtractableModel(nn.Module, ABC):

    def __init__(self):
        super().__init__()
        self.classifier: nn.Linear

    @abstractmethod
    def forward(
        self, x: torch.Tensor, extract_last_layer: bool = False
    ) -> torch.Tensor:
        pass


class FlexibleLeNet(ExtractableModel):
    """
    A flexible version of the LeNet architecture which allows changing the size of the model easily.
    To get the original TinyLeNet, set size_scale=0.5
    To get the original LeNet, set size_scale=1.0.

    Any value below 0.35 will behave identical to 0.35. Increments of 0.05 are ideal.
    """

    def __init__(self, num_classes: int, size_scale: float):

        super(FlexibleLeNet, self).__init__()

        kernel1_size = 5
        if size_scale < 0.5:
            kernel1_size = 3

        kernel2_size = 3
        if size_scale > 0.5:
            kernel2_size = 5

        conv1_channels = max(1, round(4 * size_scale))

        if size_scale > 0.5:
            conv2_channels = max(1, round(20 * size_scale - 8))
        else:
            conv2_channels = max(1, round(4 * size_scale))

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, conv1_channels, kernel_size=kernel1_size),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(conv1_channels, conv2_channels, kernel_size=kernel2_size),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )

        # Simplest way to calculate the number of features in the last layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 28, 28)
            dummy_output = self.feature_extractor(dummy_input)
            flat_features = dummy_output.view(1, -1).size(1)

        self.classifier = nn.Linear(flat_features, num_classes, bias=True)

    def forward(self, x, extract_last_layer=False):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # Shape: (batch_size, num_features)
        if extract_last_layer:
            return x
        x = self.classifier(x)
        return x
