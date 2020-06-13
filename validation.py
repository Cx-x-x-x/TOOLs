import torch
import torchvision
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from myfolder import myImageFolder
from torch.utils.data import DataLoader
import numpy as np
import sklearn.metrics

from cx_model.alexnet import alexnet
from cx_model.vgg import vgg16
from cx_model.resnet import resnet50
from arl_0 import ARL, BasicBlock, Block

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.72033167, 0.4602297, 0.38352215], [0.22272113, 0.19686753, 0.19163243])])
test_dataset = myImageFolder('/Disk1/chenxin/LSID3_5_1/test0', transform=test_transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=1,
                         shuffle=False, num_workers=2)

""" extract img_name list"""
# # myImageFolder的用途在这里，oneset[0]就是 __getitem__里的path
# list_test0 = []
# with open('/Disk1/chenxin/list_test0.txt', "w") as f:
#     for idx, oneset in enumerate(test_dataset):
#         f.write('%03d | %s' % (idx, oneset[0]))
#         f.write('\n')


# todo load the model
# model = ARL(BasicBlock, [2, 2, 2, 2]).to(device)

# model = vgg16().to(device)
# cl_feature = model.classifier[6].in_features
# model.classifier[6] = nn.Linear(cl_feature, 3).to(device)

model = torchvision.models.googlenet(pretrained=False, aux_logits=False).to(device)
# model = resnet50().to(device)
fc_feature = model.fc.in_features
model.fc = nn.Linear(fc_feature, 3).to(device)

model.load_state_dict(torch.load('/Disk1/chenxin/model/model_113/net_005.pth'), strict=True)


# test
a = 0
predicteds = np.array([])
labels = np.array([])
with open('/Disk1/chenxin/googlenet_result_details.txt', 'w') as fw:  # todo
    model.eval()
    for i, data in enumerate(test_loader):
        image, label = data[1], data[2]  # 因为改成了myImageFolder, 传入到loader里面也变了，原为 image, label = data
        image, label = image.to(device), label.to(device)
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        predicted_copy = predicted.data.cpu().numpy()
        label_copy = label.data.cpu().numpy()
        labels = np.append(labels, label_copy)
        predicteds = np.append(predicteds, predicted_copy)
        if label != predicted:
            fw.write('%4d | %d | %d' % (i, label_copy, predicted_copy))
            fw.write('\n')

            # print('index = ', i)
            # print('label = ', labels)
            # print('predition = ', predicted)
            # print('output = ', outputs)
            a += 1

error = a / len(test_loader) * 100
print('error = ', error)

with open('/Disk1/chenxin/googlenet_score.txt', 'w') as fs:  # todo
    recall = sklearn.metrics.recall_score(labels, predicteds, average=None)
    precision = sklearn.metrics.precision_score(labels, predicteds, average=None)
    avg_recall = sklearn.metrics.recall_score(labels, predicteds, average='weighted')
    avg_precision = sklearn.metrics.precision_score(labels, predicteds, average='weighted')
    acc = sklearn.metrics.accuracy_score(labels, predicteds)
    f1_score = sklearn.metrics.f1_score(labels, predicteds, average=None)
    avg_f1_score = sklearn.metrics.f1_score(labels, predicteds, average='weighted')

    fs.write('recall: %.2f, %.2f, %.2f, %.2f' % (recall[0], recall[1], recall[2], avg_recall))
    fs.write('\n')
    fs.write('precision: %.2f, %.2f, %.2f, %.2f' % (precision[0], precision[1], precision[2], avg_precision))
    fs.write('\n')
    fs.write('f1_score:  %.2f, %.2f, %.2f, %.2f' % (f1_score[0], f1_score[1], f1_score[2], avg_f1_score))
    fs.write('\n')
    fs.write('acc: %d%%' % (acc * 100))

# print('recall = ', recall)
# print('precision = ', precision)
# print('avg_recall = ', avg_recall)
# print('avg_precision = ', avg_precision)



