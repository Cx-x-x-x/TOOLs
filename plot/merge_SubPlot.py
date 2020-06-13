import matplotlib.pyplot as plt

"""
    将所有模型的某一种曲线几种在一起，放在subplot里
    对比哪个曲线的趋势比较好
"""

root_dir = 'C:/Users/Administrator/Desktop/Loss Acc/'

a = 1
for i in range(38, 41):  # 存储的文件夹序号
    sub_dir = str(i)
    train_fn = root_dir + sub_dir + '/train_acc_loss.txt'
    train_X, train_acc, train_loss = [], [], []
    with open(train_fn, 'r') as f1:
        lines = f1.readlines()[0:100]  # epoch
        for line in lines:
            value = [float(s.strip('%')) for s in line.split()]
            train_X.append(value[0])
            train_acc.append(value[1])
            train_loss.append(value[2])
    # plt.subplot(3, 3, a)  # gird
    plt.xlabel('epoch')
    plt.ylabel('ACC')
    plt.title(str(i))
    plt.legend
    plt.plot(train_X, train_acc, label='train_acc')  # acc or loss
    a += 1

plt.show()
