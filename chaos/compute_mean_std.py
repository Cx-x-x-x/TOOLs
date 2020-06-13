import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# todo "transforms mean and std"
transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder('/Disk1/chenxin/LSID3_5_1/train0', transform=transform)


print(dataset.classes)
print(dataset.class_to_idx)

# dataset[i][0] 是一个tensor，data[i][1]是class
print('第一张图片张量', dataset[0][0])
print('this img is class', dataset.classes[dataset[0][1]])
print('第一张图片张量的shape是', dataset[0][0].shape)

# dataset[0] 是一个 tuple,所以没有shape属性
print('dataset[0] = ', dataset[0])
# print(dataset[0].shape)

means = torch.zeros(3)
stds = torch.zeros(3)

for data in dataset: # todo note:for循环中的写法
    img = data[0]
    for i in range(3):
        means[i] += img[i, :, :].mean()
        stds[i] += img[i, :, :].std()


num_img = len(dataset)  # dataset is "a list of tuple" ==> len(dataset) is the # of images
mean = np.asarray(means) / num_img
std = np.asarray(stds) / num_img

print('mean = ', mean)
print('std = ', std)