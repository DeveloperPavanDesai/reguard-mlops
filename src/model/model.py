import torch.nn as nn

class ANN(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super(ANN, self).__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)