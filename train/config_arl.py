import torch
from arl_0 import Block, BasicBlock

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SaveFreq = 5
save_dir = 'model_115/'
tensorboard_dir = '115/'

# /Disk1/chenxin/model/model_73/net_040.pth
# /Disk1/chenxin/model/resnet50-19c8e357.pth
pthfile = '/Disk1/chenxin/model/resnet18-5c106cde.pth'

block = BasicBlock  # 不能 BasicBlock(add_softmax='spatial')  # add_dropout='conv2'
layers = [2, 2, 2, 2]
dropout = 'fc'

factor_init = 0.01

Optimizer = 'adam'
lr = 0.001
BatchSize = 64
wd = 7.11E-04

Epoch = 50



