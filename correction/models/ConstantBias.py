from torch import nn
import torch.nn.functional as F
import torch


class ConstantBias(nn.Module):
    def __init__(self, channels=6):
        super().__init__()
        self.channels = channels
        self.linear = nn.Linear(channels*210*280, 3)

    def forward(self, x):
        '''
        input: S*B*C*H*W
        :param input:
        :return:
        '''
        # print(x.min().item(), x.max().item(), x.mean().item(), x.std().item(), 'x min max')
        output = self.linear(x.view(*x.shape[:-3], self.channels*210*280))
        # print(output.min().item(), output.max().item(), output.mean().item(), output.std().item(), 'c min max')
        o_input = torch.split(x, 3, dim=-3)
        output = o_input[0].permute(3, 4, 0, 1, 2) + output
        return output.permute(2, 3, 4, 0, 1)
