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
    # 1.初始化父类成员，搭建神经网络
    def __init__(self):
        # 1.1初始化父类成员
        super().__init__()
        # 1.2搭建神经网络
        # 第1个卷积层，输入2通道，输出16通道，卷积核1*3，步长1，填充0
        self.conv1 = nn.Conv1d(2,16,3,1,0)
        # 第1个平均池化层，窗口 1*2，步长2，填充0
        self.pool1 = nn.AvgPool1d(2,1,0)

        # 第2个卷积层，输入6通道，输出16通道，卷积核大小1*3，步长1，填充0
        self.conv2 = nn.Conv1d(16,128,5,1,0)
        # 第2个池化层，窗口 1*2，步长1，填充0
        self.pool2 = nn.AvgPool1d(2,1,0)

        # shortcut
        self.shortcut = nn.Conv1d(2,128,1,5,0)

        # 第1个隐藏层(全连接层),输入64,输出128
        self.linear1 = nn.Linear(256,512)
        # 第2个隐藏层(全连接层),输入128,输出64
        self.linear2 = nn.Linear(512,256)
        # 第3个隐藏层(全连接层),输入128,输出64
        self.linear3 = nn.Linear(256,128)
        # 第4个隐藏层(全连接层),输入64,输出4
        self.output = nn.Linear(128,4)


    # 2.定义前向传播
    def forward(self, x):
        residual = self.shortcut(x)

        # 第1层:卷积(加权求和) + 激活函数 --> 池化(降维)
        x = self.pool1(torch.relu(self.conv1(x)))

        # 第2层:卷积(加权求和) + 激活函数 --> 池化(降维)
        x = self.pool2(torch.relu(self.conv2(x)))
        # 参1：样本行数，参2：特征列数，-1表示自动计算

        # 残差
        x += residual

        x = x.reshape(x.size(0),-1)     # n行48列，一行是一个数据
        # 第3层：全连接层(加权求和)+激活函数
        x = torch.relu(self.linear1(x))

        # 第4层：全连接层(加权求和)+激活函数
        x = torch.relu(self.linear2(x))

        # 第5层：全连接层(加权求和)+激活函数
        x = torch.relu(self.linear3(x))

        # 第6层：全连接层(加权求和)+激活函数
        x = self.output(x)  # 输出层

        return x
