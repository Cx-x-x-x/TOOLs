import torch
from torch import nn
import torchvision
from torchvision import transforms
import numpy as np
import cv2
import torch.nn.functional as F

import torchvision.utils as vutils
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from resnet import resnet50
from arl_0 import ARL, BasicBlock

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# data
test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.72033167, 0.4602297, 0.38352215], [0.22272113, 0.19686753, 0.19163243])])
test_dataset = ImageFolder('/Disk1/chenxin/LSID_error', transform=test_transform)  # '/Disk1/chenxin/LSID3_5_1/test0'
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=20)

# model
net = ARL(BasicBlock, [2, 2, 2, 2]).to(device)
pthfile = '/Disk1/chenxin/model/model_103/net_030.pth'

# net = resnet50(pretrained=False).to(device)
# pthfile = '/Disk1/chenxin/model/model_24/net_050.pth'
# fc_feature = net.fc.in_features
# net.fc = nn.Linear(fc_feature, 3).to(device)

net.load_state_dict(torch.load(pthfile))

features_blobs = []
# hook the feature extractor
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())


# get the softmax weight
params = list(net.parameters())
weight_softmax = np.squeeze(params[-2].data.cpu().numpy())


def returnCAM(feature_conv, weight_softmax):
    # generate the class activation maps upsample to 224X224
    size_upsample = (224, 224)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in range(3):  # for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


mean = [0.72033167, 0.4602297, 0.38352215]
std = [0.22272113, 0.19686753, 0.19163243]


net.eval()
# extract the image in one batch
for i, data in enumerate(test_loader):
    imgs, labels = data
    imgs, labels = imgs.to(device), labels.to(device)
    etr = i

    net.layer4.register_forward_hook(hook_feature)  # 需要在 logit=net(imgs) 前面
    logit = net(imgs)

    # h_x = F.softmax(logit, dim=1).data.squeeze()
    # probs, idx = h_x.sort(dim=0, descending=True)
    # probs = probs.cpu().numpy()
    # idx = idx.cpu().numpy()

    # generate class activation mapping for the top1 prediction
    CAMs = returnCAM(features_blobs[i], weight_softmax)

    img = imgs.squeeze(0)
    img[0] = img[0] * std[0] + mean[0]
    img[1] = img[1] * std[1] + mean[1]
    img[2] = img[2] * std[2] + mean[2]
    img = img.mul(255)

    img = img.cpu().numpy().transpose((1, 2, 0))  # C H W -> H W C
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite('/Disk1/chenxin/cam/' + str(etr) + '_cam.jpg', result)
    cv2.imwrite('/Disk1/chenxin/cam/' + str(etr) + '_org.jpg', img)
