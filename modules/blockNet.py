import torch
import torch.nn as nn

# TODO: try residual

class ResBlock(nn.Module):
    def __init__(self, in_dim, leaky_value=0.01):
        super(ResBlock, self).__init__()
        self.in_dim = in_dim # bins ** 3
        self.leaky_value = leaky_value

        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv1d(1, 16, kernel_size=3, padding=1),  # bs x 16 x in_dim
            torch.nn.BatchNorm1d(16),
            torch.nn.LeakyReLU(self.leaky_value),  # bs x 16 x in_dim
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv1d(16, 32, kernel_size=3, padding=1),  # bs x 32 x in_dim
            torch.nn.BatchNorm1d(32),
            torch.nn.LeakyReLU(self.leaky_value),  # bs x 32 x in_dim
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(32 * self.in_dim, 64 * self.in_dim),
            torch.nn.ReLU()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x.view(x.shape[0], -1)) # bs x 64 * self.in_dim
        return x, torch.sigmoid(x)


if __name__ == '__main__':
    x = torch.randn(32, 1, 1000)
    bn = ResBlock(1000)
    out, p = bn(x)
