import torch
import torch.nn as nn

class SemanticExtractor(nn.Module):
      def __init__(self):
                super(SemanticExtractor, self).__init__()
                # Conv block 1
                self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
                self.relu1 = nn.ReLU()
                self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 2
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.relu2 = nn.ReLU()
                self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 3
                self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                self.relu3 = nn.ReLU()
                self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
                self.fc1 = nn.Linear(128 * 28 * 28, 256)
                self.relu4 = nn.ReLU()
                self.fc2 = nn.Linear(256, 2)

      def forward(self, x):
                x = self.pool1(self.relu1(self.conv1(x)))
                x = self.pool2(self.relu2(self.conv2(x)))
                x = self.pool3(self.relu3(self.conv3(x)))
                x = x.view(x.size(0), -1) # Flatten
        features = self.relu4(self.fc1(x))
        out = self.fc2(features)
        return out, features
