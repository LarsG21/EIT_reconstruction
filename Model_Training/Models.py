import torch
import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, output_size),
            # nn.Sigmoid(),  # Sigmoid activation to ensure pixel values between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class LinearModelWithDropout(nn.Module):
    def __init__(self, input_size, output_size, dropout_prob=0.1):
        super(LinearModelWithDropout, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),  # Adding dropout after the first layer
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),  # Adding dropout after the second layer
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),  # Adding dropout after the third layer
        )
        self.decoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),  # Adding dropout after the first decoder layer
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),  # Adding dropout after the second decoder layer
            nn.Linear(128, output_size)
            # nn.Sigmoid(),  # Sigmoid activation to ensure pixel values between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class LinearModelWithDropout2(nn.Module):
    def __init__(self, input_size, output_size, dropout_prob=0.1):
        super(LinearModelWithDropout2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),  # Adding dropout after the first layer
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),  # Adding dropout after the second layer
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),  # Adding dropout after the third layer
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),  # Adding dropout after the first decoder layer
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),  # Adding dropout after the second decoder layer
            nn.Linear(128, output_size)
            # nn.Sigmoid(),  # Sigmoid activation to ensure pixel values between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class LinearModelWithDropout2_less_deep(nn.Module):
    def __init__(self, input_size, output_size, dropout_prob=0.1):
        super(LinearModelWithDropout2_less_deep, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),  # Adding dropout after the first layer
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),  # Adding dropout after the second layer
        )
        self.decoder = nn.Sequential(
            nn.Dropout(dropout_prob),  # Adding dropout after the first decoder layer
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),  # Adding dropout after the second decoder layer
            nn.Linear(128, output_size)
            # nn.Sigmoid(),  # Sigmoid activation to ensure pixel values between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class LinearModelWithDropoutAndBatchNorm(nn.Module):
    def __init__(self, input_size, output_size, dropout_prob=0.1):
        super(LinearModelWithDropoutAndBatchNorm, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),  # Batch Normalization after the first encoder layer
            nn.ReLU(True),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),  # Batch Normalization after the second encoder layer
            nn.ReLU(True),
            nn.Dropout(dropout_prob),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),  # Batch Normalization after the third encoder layer
            nn.ReLU(True),
            nn.Dropout(dropout_prob)
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),  # Batch Normalization after the first decoder layer
            nn.ReLU(True),
            nn.Dropout(dropout_prob),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),  # Batch Normalization after the second decoder layer
            nn.ReLU(True),
            nn.Dropout(dropout_prob),
            nn.Linear(128, output_size)
            # nn.Sigmoid(),  # You can add Sigmoid activation if needed
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# TODO: Implement the ConvolutionalModel class
class ConvolutionalModelWithDropout(nn.Module):
    def __init__(self, input_size, output_size, dropout_prob=0.1):
        super(ConvolutionalModelWithDropout, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),  # Adding dropout after the first convolutional layer
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),  # Adding dropout after the second convolutional layer
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),  # Adding dropout after the third convolutional layer
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),  # Adding dropout after the first decoder convolutional layer
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),  # Adding dropout after the second decoder convolutional layer
            nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=3, padding=1),
            nn.Linear(in_features=896, out_features=output_size)  # Adjust the input size if necessary
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ConvolutionalModelWithDecoder(nn.Module):
    def __init__(self, input_size, output_size, dropout_prob=0.1):
        super(ConvolutionalModelWithDecoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=3, padding=1),
            nn.ReLU(True),  # You can use ReLU or any other activation based on your task
        )

        # Additional layers for reshaping the input into a 2D shape
        # self.reshape_layer = nn.Linear(input_size, 64 * initial_length)  # Adjust initial_length according to your data

    def forward(self, x):
        # Reshape input to 2D
        # x = self.reshape_layer(x)
        # x = x.view(x.size(0), 64, -1)  # Assuming 64 channels, adjust if needed

        x = self.encoder(x)
        x = self.decoder(x)
        return x




if __name__ == '__main__':
    # Example usage:
    # Assuming input has 64 channels and you want the output size to be 4096
    model = ConvolutionalModelWithDecoder(input_size=1024, output_size=4096)
    input_data = torch.randn(1, 64, 1024)  # Adjust your_input_length according to your data

    # Forward pass
    output = model(input_data)

    # Check the output shape
    print(output.shape)
