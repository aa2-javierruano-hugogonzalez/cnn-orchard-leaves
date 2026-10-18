import torch
import torch.nn as nn


def count_flattened_features(feature_extractor, input_size):
    dummy_batch = torch.zeros(1, 3, input_size, input_size)
    feature_extractor.eval()
    with torch.no_grad():
        flattened_features = torch.flatten(feature_extractor(dummy_batch), start_dim=1).size(1)
    feature_extractor.train()
    return flattened_features


class BaselineCNN(nn.Module):
    def __init__(self, num_classes=7, conv_filters=(16, 32, 64), kernel_size=5, fc_width=128,
                 dropout_rate=0.25, activation=nn.ReLU, use_batchnorm=False, use_global_pooling=False,
                 input_size=96):
        super().__init__()

        convolutional_layers = []
        in_channels = 3
        for out_channels in conv_filters:
            convolutional_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1,
                                                  padding=kernel_size // 2, bias=not use_batchnorm))
            if use_batchnorm:
                convolutional_layers.append(nn.BatchNorm2d(out_channels))
            convolutional_layers.append(activation())
            convolutional_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels

        self.features = nn.Sequential(*convolutional_layers)

        if use_global_pooling:
            self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
            classifier_in_features = in_channels
        else:
            self.global_pool = nn.Identity()
            classifier_in_features = count_flattened_features(self.features, input_size)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=classifier_in_features, out_features=fc_width),
            activation(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=fc_width, out_features=num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
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
