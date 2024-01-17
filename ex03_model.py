import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ShallowCNN(nn.Module):
    def __init__(self, hidden_features=32, num_classes=42, **kwargs):
        super().__init__()
        c_hid1 = hidden_features
        c_hid2 = hidden_features * 2
        self.c_hid3 = hidden_features * 4

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, c_hid1, kernel_size=5, stride=2, padding=4),
            Swish(),
            nn.Conv2d(c_hid1, c_hid2, kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(c_hid2, self.c_hid3, kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(self.c_hid3, self.c_hid3, kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(self.c_hid3, self.c_hid3, kernel_size=3, stride=2, padding=1),
            Swish(),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.c_hid3, num_classes)
        )

    def get_logits(self, x):
        return self.fc_layers(F.adaptive_avg_pool2d(self.cnn_layers(x), 1))

    def forward(self, x, y=None) -> torch.Tensor:
        x = self.get_logits(x)
        if y is None:
            return torch.logsumexp(x, dim=1)
        else:
            return torch.exp(x)[:, y]