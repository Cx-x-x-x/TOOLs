import matplotlib.pyplot as plt

"""
      将一个模型的train_acc和test_acc曲线合并，train_loss和test_loss曲线合并
      观察过拟合的情况  
"""

# convenience
root_dir = 'C:/Users/Administrator/Desktop/LZX/Loss Acc/'
sub_dir = '1/'  # 选择存储的文件夹


# plot
# train_fn = 'C:/Users/Administrator/Desktop/Loss Acc/7/train_acc_loss.txt'
train_fn = root_dir + sub_dir + 'train_acc_loss.txt'
train_X, train_acc, train_loss = [], [], []
with open(train_fn, 'r') as f1:
    lines = f1.readlines()[0:100]  # 选择画到第几epoch
    for line in lines:
        value = [float(s.strip('%')) for s in line.split()]
        train_X.append(value[0])
        train_acc.append(value[1])
        train_loss.append(value[2])

# test_fn = 'C:/Users/Administrator/Desktop/Loss Acc/7/test_acc_loss.txt'
test_fn = root_dir + sub_dir + 'test_acc_loss.txt'
test_X, test_acc, test_loss = [], [], []
with open(test_fn, 'r') as f2:
    lines = f2.readlines()[0:100]  # 选择画到第几epoch
    for line in lines:
        value = [float(s.strip('%')) for s in line.split()]
        test_X.append(value[0])
        test_acc.append(value[1])
        test_loss.append(value[2])


plt.plot(train_X, train_acc, label='train_acc')
plt.plot(test_X, test_acc, label='test_acc')
plt.legend(loc='upper left')
# plt.savefig('C:/Users/Administrator/Desktop/Loss Acc/7/acc.png', bbox_inches='tight')
plt.savefig(root_dir + sub_dir + 'acc.png', bbox_inches='tight')
plt.show()


plt.plot(train_X, train_loss, label='train_loss')
plt.plot(test_X, test_loss, label='test_loss')
plt.legend(loc='upper right')
# plt.savefig('C:/Users/Administrator/Desktop/Loss Acc/7/loss.png', bbox_inches='tight')
plt.savefig(root_dir + sub_dir + 'loss.png', bbox_inches='tight')
plt.show()
