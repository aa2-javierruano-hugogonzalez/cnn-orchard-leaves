import torch.nn as nn


class LeafCNN(nn.Module):
    def __init__(self, num_classes=7, in_channels=3, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            self._bloque(in_channels, 32),
            self._bloque(32, 64),
            self._bloque(64, 128),
            self._bloque(128, 256),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _bloque(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(num_classes=7, in_channels=3, dropout=0.3, device=None):
    model = LeafCNN(num_classes=num_classes, in_channels=in_channels, dropout=dropout)
    if device is not None:
        model = model.to(device)
    return model