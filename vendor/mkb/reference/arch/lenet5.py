import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_classes):
        """
        LeNet-5 architecture implementation in PyTorch.

        :param num_classes: The number of output classes.
        """
        super(Model, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x, conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b):
        """
        Forward pass of the LeNet-5 model with all weights as inputs.

        :param x: The input tensor, shape (batch_size, 1, 32, 32)
        :param conv1_w, conv1_b: First conv layer weights and bias
        :param conv2_w, conv2_b: Second conv layer weights and bias
        :param fc1_w, fc1_b: First fc layer weights and bias
        :param fc2_w, fc2_b: Second fc layer weights and bias
        :param fc3_w, fc3_b: Third fc layer weights and bias
        :return: The output tensor, shape (batch_size, num_classes)
        """
        # First convolutional layer with ReLU activation and max pooling
        x = F.relu(F.conv2d(x, conv1_w, conv1_b, stride=1))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Second convolutional layer with ReLU activation and max pooling
        x = F.relu(F.conv2d(x, conv2_w, conv2_b, stride=1))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Flatten the output for the fully connected layers
        x = x.view(-1, 16*5*5)

        # First fully connected layer with ReLU activation
        x = F.relu(F.linear(x, fc1_w, fc1_b))

        # Second fully connected layer with ReLU activation
        x = F.relu(F.linear(x, fc2_w, fc2_b))

        # Final fully connected layer
        x = F.linear(x, fc3_w, fc3_b)

        return x

# Test code for the LeNet-5 model (larger batch & image)
batch_size = 4096
num_classes = 20

def get_inputs():
    x = torch.rand(batch_size, 1, 32, 32)
    conv1_w = torch.rand(6, 1, 5, 5)
    conv1_b = torch.rand(6)
    conv2_w = torch.rand(16, 6, 5, 5)
    conv2_b = torch.rand(16)
    fc1_w = torch.rand(120, 16*5*5)
    fc1_b = torch.rand(120)
    fc2_w = torch.rand(84, 120)
    fc2_b = torch.rand(84)
    fc3_w = torch.rand(num_classes, 84)
    fc3_b = torch.rand(num_classes)
    return [x, conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b]

def get_init_inputs():
    return [num_classes]