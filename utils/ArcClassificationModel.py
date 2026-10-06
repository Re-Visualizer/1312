"""
这里面是硬编写的各类神经网络的参数

"""
import torch
import torch.nn as nn

# 浅层全连接神经网络
class DNN(nn.Module):
    # 1.在init魔法方法中，初始化父类成员，及搭建神经网络
    def __init__(self, input_dim, output_dim):
        # 1.1初始化父类成员
        super().__init__()
        # 1.2搭建神经网络
        # 隐藏层1
        self.linear1 = nn.Linear(input_dim, 64)
        # 隐藏层2
        self.linear2 = nn.Linear(64,256)
        # 输出层
        self.output = nn.Linear(256, output_dim)

    # 2.前向传播forward()
    def forward(self, x):
        # 2.1隐藏层1：加权求和 + 激活函数Relu
        x = torch.relu(self.linear1(x))
        # 2.2隐藏层2：加权求和 + 激活函数Relu
        x = torch.relu(self.linear2(x))

        # 2.3输出层：加权求和
        # x = torch.softmax(self.output(x),dim=1)#正常写法，但不需要，后续用多分类交叉熵损失函数CrossEntropLoss()替代
        # CrossEntropLoss() = softmax() + 损失计算
        x = self.output(x)
        # 2.4返回处理结果
        return x

# 卷积神经网络 + 残差网络
class CNN(nn.Module):
    """
    a designed fixed Convolutional Neural Network class for 1D data
    formula: len = (len + 2*padding - kernel_size) / stride + 1
    """
    # 1.初始化父类成员，搭建神经网络
    def __init__(self):
        # 1.1初始化父类成员
        super().__init__()

        # 1.2搭建神经网络
        # 第1个卷积层，输入通道，输出通道，卷积核，步长，填充
        self.conv1 = nn.Conv1d(2,32,3,1,0)
        # 第1个最大池化层，窗口，步长，填充
        self.pool1 = nn.MaxPool1d(2,1,0)

        # 第2个卷积层
        self.conv2 = nn.Conv1d(32,128,3,1,0)
        # 第2个池化层
        self.pool2 = nn.MaxPool1d(2,1,0)

        # shortcut
        self.shortcut1 = nn.Conv1d(2,128,4,1,0)
        # 第3个池化层
        self.pool3 = nn.MaxPool1d(4,1,0)

        # 第1个隐藏层(全连接层)
        self.linear1 = nn.Linear(512,384)

        # 第2个隐藏层(全连接层)
        self.linear2 = nn.Linear(384,256)

        # 第3个隐藏层(全连接层)
        self.linear3 = nn.Linear(256,128)

        # shortcut
        self.shortcut2 = nn.Linear(512,128)

        # 第4个隐藏层(全连接层)
        self.output = nn.Linear(128,4)



    # 2.定义前向传播
    def forward(self, x):
        residual1 = self.pool3(torch.relu(self.shortcut1(x)))

        # 第1层:卷积(加权求和) + 激活函数 --> 池化(降维)
        x = self.pool1(torch.relu(self.conv1(x)))

        # 第2层:卷积(加权求和) + 激活函数 --> 池化(降维)
        x = self.pool2(torch.relu(self.conv2(x)))
        # 参1：样本行数，参2：特征列数，-1表示自动计算

        # 残差1
        x = x + residual1

        # reshape展平特征为一维
        x = x.reshape(x.size(0),-1)

        residual2 = torch.relu(self.shortcut2(x))
        # 第3层：全连接层(加权求和)+激活函数
        x = torch.relu(self.linear1(x))

        # 第4层：全连接层(加权求和)+激活函数
        x = torch.relu(self.linear2(x))

        # 第5层：全连接层(加权求和)+激活函数
        x = torch.relu(self.linear3(x))

        # 残差2
        x = x + residual2

        # 第6层：全连接层(加权求和)+激活函数
        x = self.output(x)  # 输出层

        return x
