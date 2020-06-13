import torch
from torch import nn
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from cx_model.alexnet import alexnet
from tensorboardX import SummaryWriter


writer = SummaryWriter('/Disk1/chenxin/runs/alexnet')  # todo log file

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AlexNet(nn.Module):
    def __init__(self, pretrained=True):
        super(AlexNet, self).__init__()
        self.net = alexnet(pretrained).features.eval()

    def forward(self, x):
        out = []
        for i in range(len(self.net)):
            x = self.net[i](x)
            if i in [10]:  # todo select the extracted layers in []
                print(self.net[i])  # todo delete
                out = x  # todo xiugai
        return out


# data
test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.72033167, 0.4602297, 0.38352215], [0.22272113, 0.19686753, 0.19163243])])
test_dataset = ImageFolder('/Disk1/chenxin/LSID3_5_1/test0', transform=test_transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False, num_workers=20)

# model
model = AlexNet().to(device)


# extract images in one batch
for i, data in enumerate(test_loader):
    if i == 0:
        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)

# get feature map
feature_map = model(imgs)
print('feature_map.shape =', feature_map.shape)

# the 1st image in batch
print('img.shape = ', imgs.shape)
# imgs = imgs[0].detach().cpu().unsqueeze(dim=1)
imgs = imgs[0].detach().cpu()
print('trans_img.shape = ', imgs.shape)
img_grid = vutils.make_grid(imgs, normalize=True)  # todo -normalization
writer.add_image('original_map', img_grid)

feature_map = feature_map[0].detach().cpu().unsqueeze(dim=1)  # [C, 1, H, W]
print('trans_feature.shape = ', feature_map.shape)
feature_grid = vutils.make_grid(feature_map, normalize=True, nrow=8)  # todo -normalization
writer.add_image('layer[0]' + '_feature_maps', feature_grid)

writer.close()
