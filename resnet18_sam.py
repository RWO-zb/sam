import torch
import torch.nn as nn
from torchvision import models

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)

class ResNet18_SAM(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base = models.resnet18(weights=None)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.bn = nn.BatchNorm1d(512, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256, eps=1e-3),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self.apply(init_weights)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc(x)
        return x