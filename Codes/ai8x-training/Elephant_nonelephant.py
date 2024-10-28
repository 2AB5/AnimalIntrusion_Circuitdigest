"""
Elephant vs Non-Elephant CNN for MAX78000FTHR.

A simple convolutional neural network designed for the classification of elephant vs non-elephant
images. The model is optimized for deployment on edge devices like the MAX78000FTHR.

"""

from torch import nn
import ai8x

class ElephantCNN(nn.Module):
    """
    CNN Model for Elephant vs Non-Elephant Classification
    """
    def __init__(self, num_classes=2, num_channels=3, dimensions=(64, 64), bias=False, **kwargs):
        super().__init__()

        # First convolutional layer (3x3), 16 filters
        self.conv1 = ai8x.FusedConv2dBNReLU(num_channels, 16, 3, stride=1, padding=1, bias=bias, **kwargs)

        # Max-Pooling Layer (2x2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Second convolutional layer (3x3), 32 filters
        self.conv2 = ai8x.FusedConv2dBNReLU(16, 32, 3, stride=1, padding=1, bias=bias, **kwargs)

        # Max-Pooling Layer (2x2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layer with 8192 inputs (32x16x16), 128 outputs
        self.fc = nn.Linear(32 * 16 * 16, 128, bias=bias)

        # Output layer (128 inputs, 2 outputs for binary classification)
        self.output = nn.Linear(128, num_classes, bias=bias)

    def forward(self, x):
        """Forward propagation"""
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        x = self.output(x)
        return x

def elephant_cnn(pretrained=False, **kwargs):
    """
    Constructs the CNN model for elephant vs non-elephant classification.
    """
    assert not pretrained
    return ElephantCNN(**kwargs)

models = [
    {
        'name': 'elephant_cnn',
        'min_input': 1,
        'dim': 2,
    },
]
