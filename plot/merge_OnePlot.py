import matplotlib.pyplot as plt

"""
    将多个模型的 train_acc 或 train_loss 或 test_acc 或 test_loss
    放在同一张图里比较 
"""

# convenience
root_dir = 'C:/Users/Administrator/Desktop/Loss Acc/'
sub_dir1 = '112/'  # todo
sub_dir2 = '105/'
sub_dir3 = '108/'
type_dir = 'test_acc_loss.txt'


fn1 = root_dir + sub_dir1 + type_dir
X1, acc1, loss1 = [], [], []
with open(fn1, 'r') as f1:
    lines = f1.readlines()[0:100]  # todo
    for line in lines:
        value = [float(s.strip('%')) for s in line.split()]
        X1.append(value[0])
        acc1.append(value[1])
        loss1.append(value[2])

fn2 = root_dir + sub_dir2 + type_dir
X2, acc2, loss2 = [], [], []
with open(fn2, 'r') as f2:
    lines = f2.readlines()[0:100]  # todo
    for line in lines:
        value = [float(s.strip('%')) for s in line.split()]
        X2.append(value[0])
        acc2.append(value[1])
        loss2.append(value[2])

fn3 = root_dir + sub_dir3 + type_dir
X3, acc3, loss3 = [], [], []
with open(fn3, 'r') as f3:
    lines = f3.readlines()[0:100]  # todo
    for line in lines:
        value = [float(s.strip('%')) for s in line.split()]
        X3.append(value[0])
        acc3.append(value[1])
        loss3.append(value[2])


plt.xlabel('epoch')
plt.ylabel('ACC')
plt.plot(X1, acc1, label='spatial softmax')
plt.plot(X2, acc2, label='channel softmax')
plt.plot(X3, acc3, label='mixed softmax')
plt.legend(loc='lower right')
# plt.savefig(root_dir + sub_dir + 'acc.png', bbox_inches='tight')
plt.show()

