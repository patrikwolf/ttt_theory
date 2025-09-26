import torch.nn as nn

class TorchLinearClassifier(nn.Module):
    def __init__(self, input_features, num_classes):
        super(TorchLinearClassifier, self).__init__()
        self.layer = nn.Linear(input_features, num_classes)

    def forward(self, x):
        return self.layer(x)