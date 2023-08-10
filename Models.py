import torch
import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
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
