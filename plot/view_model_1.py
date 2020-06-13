import torch
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from arl_basic import ARL, BasicBlock
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter('/Disk1/chenxin/runs/arl')

model = ARL(BasicBlock, [2, 2, 2, 2]).to(device)


writer.add_graph(model, torch.randn([64, 3, 224, 224]).to(device))
writer.close()