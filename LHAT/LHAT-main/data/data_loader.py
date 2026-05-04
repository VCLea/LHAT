import os
from tkinter.messagebox import RETRY

import pandas as pd
import torch


def load_ft(data_folder, omics_list, dataDir):
    """
    加载多组学数据
    """
    data_folder = os.path.join(data_folder, dataDir)  # 拼接完整数据路径
    label = pd.read_csv(data_folder + '/labels.csv', header=None)  # 读取标签文件，无表头
    label_item = torch.LongTensor(label.values)  # 将标签转换为LongTensor
    cuda = True if torch.cuda.is_available() else False  # 检查CUDA可用性（虽然不再使用）

    data_ft_list = []  # 初始化原始数据列表
    for i in range(len(omics_list)):
        # 读取每个组学的CSV文件并转换为numpy数组
        data_ft_list.append((pd.read_csv(os.path.join(data_folder, omics_list[i] + ".csv")).values))

    data_tensor_list = []  # 初始化张量列表
    for i in range(len(data_ft_list)):
        # 将每个组学数据转换为FloatTensor
        data_tensor_list.append(torch.FloatTensor(data_ft_list[i]))
        # 删除原代码中的cuda条件判断和.cuda()调用

    # 删除原代码中的cuda条件判断和.cuda()调用
    return data_tensor_list, label_item.reshape(-1)  # 返回张量列表和重塑为一维的标签