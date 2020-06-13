import torch
from torch import nn
import numpy as np

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.nn.functional as F

from tensorboardX import SummaryWriter
# from cx_model.resnet import resnet50
from cx_model.alexnet import alexnet

writer = SummaryWriter('/Disk1/chenxin/runs/test')  # todo log file
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# '''resnet50'''
# net = resnet50().to(device)
# fc_feature = net.fc.in_features
# net.fc = nn.Linear(fc_feature, 3).to(device)
# pretrained_dict = torch.load('/Disk1/chenxin/net_050.pth')  # todo 从本地上传历史pth
# net.load_state_dict(pretrained_dict)
'''alexnet'''
net = alexnet().to(device)
cl_feature = net.classifier[6].in_features
net.classifier[6] = nn.Linear(cl_feature, 3).to(device)
pretrained_dict = torch.load('/Disk1/chenxin/net_050.pth')  # todo 从本地上传历史pth
net.load_state_dict(pretrained_dict)


# data
test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.72033167, 0.4602297, 0.38352215], [0.22272113, 0.19686753, 0.19163243])])
test_dataset = ImageFolder('/Disk1/chenxin/LSID3_5_1/test0', transform=test_transform)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=128,
                         shuffle=False,
                         num_workers=20)

img, label = iter(test_loader).next()
x = img.to(device)
view_layer = 'features'

for name, layer in net._modules.items():

    # 为fc层预处理x
    # x = x.view(x.size(0), -1) if "fc" in name else x
    if name == view_layer:
        x = layer(x)
        x1 = x[0].detach().cpu().unsqueeze(dim=1)  # [B, 1, H, W]
        print(x.size())

        # 由于__init__()相较于forward()缺少relu操作，需要手动增加
        x = F.relu(x) if 'conv' in name else x

        img_grid = vutils.make_grid(x1, normalize=True)  # B，C, H, W
        writer.add_image(name + '_feature_maps', img_grid)

# for name, layer in net._modules.items():
#
#     # 为fc层预处理x
#     # x = x.view(x.size(0), -1) if "fc" in name else x
#
#     # 对x执行单层运算
#     x = layer(x)
#     x1 = x[0].detach().cpu().unsqueeze(dim=1)  # [B, 1, H, W]
#     print(x.size())
#
#     # 由于__init__()相较于forward()缺少relu操作，需要手动增加
#     x = F.relu(x) if 'conv' in name else x
#
#     if name == view_layer:
#         img_grid = vutils.make_grid(x1, normalize=True)  # B，C, H, W
#         writer.add_image(name + '_feature_maps', img_grid)
#
#     # # 依据选择的层，进行记录feature maps
#     # if name == vis_layer:
#     #     # 绘制feature maps
#     #     x1 = x.transpose(0, 1)  # C，B, H, W  ---> B，C, H, W
#     #     img_grid = vutils.make_grid(x1, normalize=True, scale_each=True, nrow=2)  # B，C, H, W
#     #     writer.add_image(vis_layer + '_feature_maps', img_grid, global_step=666)
#
#         # 绘制原始图像
#         # img_raw = normalize_invert(img, normMean, normStd)  # 图像去标准化
#         # img_raw = np.array(img_raw * 255).clip(0, 255).squeeze().astype('uint8')
#         # writer.add_image('raw img', img_raw, global_step=666)  # j 表示feature map数

writer.close()
