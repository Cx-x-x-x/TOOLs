import torch
import torchvision.models as models
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# pretrained=True就可以使用预训练的模型
model = models.resnet50(pretrained=False).to(device)
fc_feature = model.fc.in_features
model.fc = nn.Linear(fc_feature, 3).to(device)
pthfile = '/Disk1/chenxin/model/model_42/net_020.pth'
model.load_state_dict(torch.load(pthfile))
print(model)

