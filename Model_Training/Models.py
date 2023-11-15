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
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Using Sigmoid activation to ensure output values are between 0 and 1
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


class UNetModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(UNetModel, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_size, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True)
        )

        # Max-pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, output_size, kernel_size=3, padding=1),
            nn.ReLU(True)
        )

    def forward(self, x):
        # Encoder
        x1 = self.encoder(x)
        x2 = self.pool(x1)

        # Decoder with skip connection
        x3 = self.decoder(x2)
        x4 = nn.functional.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)
        x5 = x4 + x1

        return x5

if __name__ == '__main__':
    # # Assuming you have VOLTAGE_VECTOR_LENGTH, OUT_SIZE defined elsewhere
    # in_channels = 1  # Assuming grayscale images
    # out_channels = 1  # Assuming grayscale output
    #
    # # Example input data for a batch of images
    # batch_size = 32
    # height = 64
    # width = 64
    # input_data = torch.randn(batch_size, in_channels, height, width)
    #
    # # Create the model
    # model = UNetModel(in_channels, out_channels)
    #
    # # Forward pass
    # output = model(input_data)
    # print(output.shape)

    # same test for CNNModel
    # Assuming you have VOLTAGE_VECTOR_LENGTH, OUT_SIZE defined elsewhere

    in_channels = 104  # Assuming grayscale images
    out_channels = 64  # Assuming grayscale output

    # Example input data for a batch of images
    batch_size = 32
    height = 64
    width = 64
    input_data = torch.randn(batch_size, in_channels)

    # Create the model
    model = LinearModel(in_channels, out_channels)

    # Forward pass
    output = model(input_data)
    print(output.shape)
