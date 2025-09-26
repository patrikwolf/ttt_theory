import torch.nn as nn

class TorchMLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout_rate=0.5):
        """
        Initializes a two-layer neural network (one hidden layer).

        Args:
            input_dim (int): Dimension of the input embeddings (e.g., 512).
            hidden_dim (int): Number of neurons in the hidden layer (e.g., 1024).
            num_classes (int): Number of output classes (e.g., 1000).
        """
        super(TorchMLPClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """The forward pass of the model."""
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x