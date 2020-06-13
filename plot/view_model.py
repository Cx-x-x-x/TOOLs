import torch
from tensorboardX import SummaryWriter
from cx_model.alexnet import alexnet
from cx_model.resnet import resnet50
from cx_model.vgg import vgg16
from torch import nn

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter('/Disk1/chenxin/runs/arl')

model = vgg16(pretrained=True).to(device)
cl_feature = model.classifier[6].in_features
model.classifier[6] = nn.Linear(cl_feature, 3).to(device)

writer.add_graph(model, torch.randn([128, 3, 224, 224]).to(device), comments='alexnet')
writer.close()