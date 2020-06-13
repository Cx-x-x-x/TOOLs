import numpy as np
import matplotlib.pyplot as plt

'''
    对两个参数生成随机数
    第一列为 lr
    第二列为
'''

# # learning rate
# lr_a = 0.001
# lr_b = 0.0001
# lr_a_log = np.log10(lr_a)
# lr_b_log = np.log10(lr_b)
#
# # weight_decay
# wd_a = 0.001
# wd_b = 0.0001
# wd_a_log = np.log10(wd_a)
# wd_b_log = np.log10(wd_b)
#
# # get parameters
# account = 40  # todo 粒子个数
# particle = np.random.rand(account, 2)
# print(particle)
# x = 10 ** (particle[:, 0] * (lr_a_log - lr_b_log) + lr_b_log)
# y = 10 ** (particle[:, 1] * (wd_a_log - wd_b_log) + wd_b_log)
#
# plt.scatter(x, y)
# plt.title('random parameters')
# plt.xlim((10 ** lr_b_log, 10 ** lr_a_log))
# plt.ylim((10 ** wd_b_log, 10 ** wd_a_log))
# plt.show()
# print(x)
# print(y)


# learning rate
a = 0.001
b = 0.0001
a_log = np.log10(a)
b_log = np.log10(b)
lr_log = (a_log - b_log) * np.random.rand(10, 1) + b_log
lr = 10 ** lr_log
print(lr)




