import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniVGGNet(nn.Module):
    def __init__(self, width, height, depth, classes):
        super(MiniVGGNet, self).__init__()

        # Assuming input shape is (depth, height, width)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=depth, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )

        # Compute the flattened feature size after conv layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, depth, height, width)
            flat_size = self.features(dummy_input).view(1, -1).shape[1]

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

