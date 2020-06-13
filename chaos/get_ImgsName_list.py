import torch
from torchvision import transforms
from chaos.myfolder import myImageFolder
from torch.utils.data import DataLoader

"""
    使用在 myfolder.py 里自己创建的类 myImageFolder，生成 '图片序号 + 图片路径'
    原来的 ImageFolder 类，无法获取图片路径
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.72033167, 0.4602297, 0.38352215], [0.22272113, 0.19686753, 0.19163243])])
test_dataset = myImageFolder('/Disk1/chenxin/LSID3_5_1/test0', transform=test_transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=1,
                         shuffle=False, num_workers=2)


# # myImageFolder的用途在这里，oneset[0]就是 __getitem__里的path
list_test0 = []
with open('/Disk1/chenxin/list_test0.txt', "w") as f:
    for idx, oneset in enumerate(test_dataset):
        f.write('%03d | %s' % (idx, oneset[0]))
        f.write('\n')
