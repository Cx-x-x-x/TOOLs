import torch
from torch import nn
from arl_0 import ARL, Block, BasicBlock

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model = ARL(BasicBlock, [2, 2, 2, 2]).to(device)

# # load pth
# pthfile = '/Disk1/chenxin/model/model_77/net_030.pth'
# model.load_state_dict(torch.load(pthfile), strict=True)
#
# for m in model.state_dict():
#     print(key)

# for m in model.parameters():
#     print(m)

# print(list(model.parameters()))

# for name, param in model.state_dict():
#     print(name)

# print(model.named_parameters())
# todo 有用
for name, param in model.named_parameters():
    print(name)


# for name, param in model.named_parameters():
#     if isinstance(param, nn.Parameter) and 'factor' in name:
#         print(param)
