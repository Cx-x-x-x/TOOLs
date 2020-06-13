import torch
from torchviz import make_dot
import torch
import tensorwatch as tw
from arl_basic import ARL, BasicBlock
from resnet import resnet18

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model = ARL(BasicBlock, [2, 2, 2, 2]).to(device)
model = resnet18().to(device)


x = torch.rand(64, 3, 224, 224)
y = model(x)
g = make_dot(y)
# g.render('basicARL_model', view=False)
g.render('resnet18_model', view=False)

# tw.draw_model(model, [64, 3, 224, 224], 'LR')
