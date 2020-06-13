import torch
from tensorboardX import SummaryWriter
from cx_model.arl import ARL, BasicBlock
from cx_model.vgg import vgg16

"""
   使用 tensorboard 显示网络结构
   需要最新的 tensorboardX 版本才能显示自己建的网络
   老版本只能显示官方提供的网络 
"""

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter('/Disk1/chenxin/runs/arl')

'''ARL'''
model = ARL(BasicBlock, [2, 2, 2, 2]).to(device)
'''VGG'''
# model = vgg16(pretrained=True).to(device)
# cl_feature = model.classifier[6].in_features
# model.classifier[6] = nn.Linear(cl_feature, 3).to(device)

writer.add_graph(model, torch.randn([128, 3, 224, 224]).to(device), comments='alexnet')
writer.close()