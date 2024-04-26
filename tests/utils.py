from torch.nn import Module
from torch import Tensor
import torch


class DatasetMock(object):
    def __init__(
        self, input_size: int = 10, output_size: int = 2, num_samples: int = 100
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.data = torch.randn(num_samples, input_size)
        self.targets = torch.randint(0, output_size, (num_samples,))

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)


class ModelMock(Module):
    def __init__(self, input_size: int = 10, output_size: int = 2):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, output_size)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(input_size, output_size)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x: Tensor):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return x
