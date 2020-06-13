import torch
from torch import nn
from torchvision import models
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from cx_model.alexnet import alexnet

import visdom

vis = visdom.Visdom(env='chenxin4')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = models.resnet50(pretrained=False).to(device)
fc_feature = net.fc.in_features
net.fc = nn.Linear(fc_feature, 3).to(device)
pthfile = '/Disk1/chenxin/model/model_24/net_050.pth'
net.load_state_dict(torch.load(pthfile))


class MyModel(nn.Module):
    def __init__(self, net):  # input the dim of output fea-map of Resnet:
        super(MyModel, self).__init__()

        back_bone = net

        # add_block = []
        # add_block += [nn.Linear(2048, 512)]
        # add_block += [nn.LeakyReLU(inplace=True)]
        # add_block = nn.Sequential(*add_block)
        # add_block.apply(weights_init_xavier)
        add_block = nn.ReLU(inplace=True)

        self.BackBone = back_bone
        self.add_block = add_block

    def forward(self, x):  # input is 2048!

        for name, midlayer in self.BackBone._modules.items():
            x = midlayer(x)
            print(name)
            if name == 'layer1':  # 取出resnet中的layer2层输出
                break

        # x = self.BackBone(input)
        x = self.add_block(x)

        return x


# data
test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.72033167, 0.4602297, 0.38352215], [0.22272113, 0.19686753, 0.19163243])])
test_dataset = ImageFolder('/Disk1/chenxin/LSID3_5_1/test0', transform=test_transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False, num_workers=20)

# model
# model = AlexNet().to(device)
model = MyModel(net).to(device)

# extract the 1st image in one batch
for i, data in enumerate(test_loader):
    if i == 0:
        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)


feature_map = model(imgs)  # [B_m,C,H,W]
feature_map = feature_map[0].detach().cpu().unsqueeze(dim=1)  # [C, 1, H, W] 这种shape才能传入make_grid,3dim会产生歧义
feature_grid = vutils.make_grid(feature_map, normalize=True, nrow=8)  # [3,H,W]

for i in range(0, 3):
    vis.heatmap(feature_grid[i], opts={'title': str(i)})



