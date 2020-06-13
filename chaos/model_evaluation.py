import torch
import torchvision
from torch import nn
from torchvision import transforms
from chaos.myfolder import myImageFolder
from torch.utils.data import DataLoader
import numpy as np
import sklearn.metrics

"""
    使用 model.eval() 模式对验证集进行测试
    生成 labels list 和 predicteds list
    使用 sklearn 自动统计各个类别的各种指标: precision, recall, accuracy, f1 score
    
    注意：
        这里使用的是 myImageFolder 而不是 ImageFolder
        改成 ImageFolder 也无妨，但是 test 中读取数据要改一下
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.72033167, 0.4602297, 0.38352215], [0.22272113, 0.19686753, 0.19163243])])
test_dataset = myImageFolder('/Disk1/chenxin/LSID3_5_1/test0', transform=test_transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=1,
                         shuffle=False, num_workers=2)

# load the model
model = torchvision.models.googlenet(pretrained=False, aux_logits=False).to(device)
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
        labels = np.append(labels, label_copy)  # labels list
        predicteds = np.append(predicteds, predicted_copy)  # predicteds list

        # 获取 predition 和 label 的详细对照，用来手动观察哪一张图片错了，序号在 get_ImgsName_list.py 中可查
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



