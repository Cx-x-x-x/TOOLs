import torch
from torch import optim
from torch import nn
from cx_model.alexnet import alexnet


class mynet(nn.Module):

    def __init__(self, model, pretrained):
        super(mynet, self).__init__()
        self.alexnet = model(pretrained)  # todo
        for p in self.parameters():
            p.requires_grad = False
        self.f = nn.Conv2d(2048, 512, 1)
        self.g = nn.Conv2d(2048, 512, 1)
        self.h = nn.Conv2d(2048, 2048, 1)
        self.softmax = nn.Softmax(-1)
        self.gamma = nn.Parameter(torch.FloatTensor([0.0]))
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.alexnet.fc = nn.Linear(2048, 10)  # todo


model = mynet(alexnet, pretrained=True)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)


# for k, v in model.named_parameters():
#     if k != 'XXX':
#          v.requires_grad = False  # 固定参数


for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)


print('# Model parameters:', sum(param.numel() for param in model.parameters()))

