import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear_QNet(nn.Module):
    """Two layers over an 11-bit view of the board. Small on purpose: the state is
    already the useful features, so there is nothing for depth to discover."""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear2(x)

    def save(self, file_name="model.pth"):
        # state_dict() with the parentheses. Without them this pickles the bound
        # method instead of the weights, and every training run is thrown away -
        # which is exactly what happened to the first version of this project.
        torch.save(self.state_dict(), file_name)

    def load(self, file_name="model.pth"):
        self.load_state_dict(torch.load(file_name, map_location="cpu"))
        self.eval()
        return self
