import torch
from torch import nn
import numpy as np

# m = nn.Softmax2d()
m = nn.Softmax(dim=1)
# you softmax over the 2nd dimension
input = torch.randn(2, 3, 2, 2)
print("input = ", input)
output = m(input)
print("output = ", output)
print(output.shape)

tmp = input.detach()
# print(tmp)
tmp[0, 0, 0, 0] = tmp[0, 0, 0, 0] / torch.sum(torch.exp(tmp), 1)[0, 0, 0]
print(torch.sum(torch.exp(tmp), 1).shape)
print(tmp[0, 0, 0, 0])

# print(torch.exp(tmp))  # todo torch.exp 和 tmp.exp的计算结果不同？？？
# print(torch.sum(tmp, 1).shape)
# print(torch.sum(tmp, 1))
