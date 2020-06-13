import os

'''
    在数据储存方式为，子文件夹对应类别时，
    为数据集生成对应的txt文件，包含：数据路径 + 标签
'''

# train_txt_path = os.path.join("..", "..", "Data", "train.txt")
# train_dir = os.path.join("..", "..", "Data", "train")

# valid_txt_path = os.path.join("..", "..", "Data", "valid.txt")
# valid_dir = os.path.join("..", "..", "Data", "valid")


def gen_txt(txt_path, img_dir):
    f = open(txt_path, 'w')

    for root, s_dirs, _ in os.walk(img_dir, topdown=True):  # 获取 train文件下各文件夹名称
        for sub_dir in s_dirs:
            i_dir = os.path.join(root, sub_dir)  # 获取各类的文件夹 绝对路径
            img_list = os.listdir(i_dir)  # 获取类别文件夹下所有png图片的路径
            for i in range(len(img_list)):
                if not img_list[i].endswith('jpg'):  # 若不是png文件，跳过
                    continue
                img_path = os.path.join(i_dir, img_list[i])
                label = img_path.split('/')[5]
                line = img_path + ' ' + label + '\n'
                f.write(line)
    f.close()


if __name__ == '__main__':
    gen_txt('/Disk1/chenxin/data_cx/data_train1.txt', '/Disk1/chenxin/LSID3_5_1/train1')
    # gen_txt(valid_txt_path, valid_dir)

