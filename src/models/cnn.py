import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.fc1 = nn.Linear(in_features=64, out_features=128)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))

        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)

        x = self.dropout(self.relu4(self.fc1(x)))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = BaselineCNN(num_classes=7)
    dummy_batch = torch.randn(4, 3, 96, 96)
    output = model(dummy_batch)
    total_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    print(model)
    print(f"Input shape: {tuple(dummy_batch.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Trainable parameters: {total_parameters}")