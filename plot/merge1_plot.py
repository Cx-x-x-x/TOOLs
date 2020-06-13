import matplotlib.pyplot as plt
from matplotlib import image

root_dir = 'C:/Users/Administrator/Desktop/Loss Acc/'

# a = 1
# for i in range(12, 27):
#     plt.subplot(4, 4, a)
#     i = str(i)
#     sub_dir = i + '/acc.png'
#     plt.title(i)
#     plt.imshow(image.imread(root_dir + sub_dir))
#     a += 1
#
# plt.show()


a = 1
for i in range(38, 41):  # file
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
