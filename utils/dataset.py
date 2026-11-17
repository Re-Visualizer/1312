import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def tensor_dataset(csv_path, test_size=0.2, random_state=23, stratify=True):
    """
    :param csv_path: 数据集路径
    :param test_size: 测试集占比
    :param random_state: 随机种子
    :param stratify: 是否分层抽样，即按原数据里面各类别的比例抽样
    :return: train_dataset:训练集
             test_dataset:测试集
             x_train.shape[1]:特征类别数
             len(np.unique(y)):标签类别数
             transfer:标准化器
    """
    # 1.加载数据集
    data = pd.read_csv(csv_path)

    # 2.获取数据集的特征和标签[特征1 特征2 ... 标签]
    x, y = data.iloc[:, 0:-1], data.iloc[:, -1] #x赋为特征，y赋为标签
    x = x.astype(np.float32) #将变量x属性转换为float32类型
    y = y.astype(np.int64) #将变量y属性转换为int64类型

    # 3.切分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size = test_size,
        random_state = random_state,
        stratify = y if stratify else None
    )

    # 4.创建标准化器
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train).astype(np.float32)
    x_test = transfer.transform(x_test).astype(np.float32)

    # 5.把数据集封装为张量数据集
    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.tensor(y_train.values))
    test_dataset = TensorDataset(torch.from_numpy(x_test), torch.tensor(y_test.values))

    # 6.返回TensorDataset结果
    return train_dataset, test_dataset, x_train.shape[1], len(np.unique(y)), transfer


def create_CNNdataset(csv_path,test_size=0.2,random_state=23,stratify=True):
    # 1.加载csv文件数据集
    data = pd.read_csv(csv_path)

    # 2.获取 x特征列 和 y标签列   [特征1_特征2 标签]
    x, y = data.iloc[:, 0:-1], data.iloc[:, -1]

    # 3.把特征列转为浮点型，标签转化为整型
    x = x.to_numpy(dtype=np.float32)
    y = y.to_numpy(dtype=np.int64)

    # 4.设置通道数量和每个通道的采样点数量，预设通道数为2，每个通道的采样点数量为10
    num_channels = 2
    seq_length = 10

    # 5.切分训练集和测试集
    # 参1：特征，参2：标签，参3：测试集所占比例，参4：随机种子，参5：参考标签的类别分布数量进行抽取数据
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None)
    # 5.0恢复两个通道
    # [N, 20] --> [N, 2, 10]
    x_train = x_train.reshape(-1, num_channels, seq_length)
    x_test = x_test.reshape(-1, num_channels, seq_length)
    # 5.1把两个通道分别排列为两列，便于分别标准化
    # [N, 2, 10] --> [N, 10, 2] --> [N*10, 2]
    train_points = x_train.transpose(0, 2, 1).reshape(-1, num_channels)
    test_points = x_test.transpose(0, 2, 1).reshape(-1, num_channels)
    # 5.2创建标准化器
    # 第一列计算xiao bo的均值和标准差
    # 第二列计算dian liu的均值和标准差
    transfer = StandardScaler()
    train_points = transfer.fit_transform(train_points)
    test_points = transfer.transform(test_points)
    # 5.3恢复CNN输入形状
    # [N*10, 2] --> [N, 10, 2] --> [N, 2, 10]
    x_train = train_points.reshape(-1, seq_length, num_channels).transpose(0, 2, 1)
    x_test = test_points.reshape(-1, seq_length, num_channels).transpose(0, 2, 1)

    # 6.把数据集封装成 张量数据集. 数据-->张量-->数据集TensorDataSet-->数据加载器DataLoader
    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.from_numpy(x_test), torch.tensor(y_test))

    # 7.返回TensorDataSet结果       len(np.unique(y)充当输出的标签数
    return train_dataset, test_dataset, len(np.unique(y)), transfer
