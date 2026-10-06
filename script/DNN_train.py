"""
案例：
    电弧状态分类

背景：
    基于电弧的2列特征，预测：停机0，正常1，单故障2，复合故障3

ANN案例的实现步骤：
    1.构建数据集
    2.搭建神经网络
    3.模型训练
    4.模型测试

调优思路：
    1.优化器：SGD-->Adam
    2.学习率：0.001-->0.0001
    3.对数据进行标准化
    4.增加网络的深度，调整每层的神经元个数
    5.调整训练的轮数
    6...
"""

import time      #时间模块
import torch
import joblib
import numpy as np
import torch.nn as nn                       #封装了神经网络的各种操作
import torch.optim as optim                 #优化器
from torchsummary import summary
from torch.utils.data import DataLoader     #数据加载器
from utils.dataset import create_DNNdataset
from utils.ArcClassificationModel import DNN

torch.manual_seed(24)

# 统一定义训练设备：有GPU用GPU，否则用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'当前使用的训练设备：{device}')

# todo 1.构建数据集
# 流程：数据-->张量-->数据集dataset-->数据加载器DataLoader
# 参1：数据集路径；参2：测试集划分；参3：随即种子；参4：是否打乱顺序
train_dataset, test_dataset, input_dim, output_dim, scaler = create_DNNdataset('../data/4分类电弧.csv', test_size=0.2, random_state=23,stratify=True)
# 参1：数据集对象，参2：每批次的大小，参3：是否打乱数据
train_loader = DataLoader(train_dataset, batch_size=96,shuffle=True)

# todo 2.搭建神经网络
# 1.使用神经网络模型类ArcClassificationModel，创建对象model
model = DNN(input_dim, output_dim).to(device)
    # 计算模型参数，打印神经网络的结构总览
    # 参1：模型对象，参2：输入数据的形状（批次大小，输入特征），每批96条，每条2列特征
summary(model, input_size=(96, input_dim), device=str(device))
# 2.定义损失函数，因为是多分类，这里用的是：多分类交叉熵损失函数
criterion = nn.CrossEntropyLoss()
    # 2.1创建优化器
optimizer = optim.Adam(model.parameters(),lr=0.0001, betas=(0.9, 0.999))

# todo 3.模型训练
# 1.定义变量，记录训练的总轮数
epochs = 50
# 2开始每轮的训练
for epoch in range(epochs):
    # 2.1定义变量，记录每次训练的损失值，训练批次数
    total_loss, batch_num = 0.0, 0
    # 2.2定义变量，表示训练开始的时间
    start = time.time()
    # 2.3开始每批次的训练
    for x,y in train_loader:
        # 2.3.0把数据迁移到GPU
        x = x.to(device)
        y = y.to(device)
        # 2.3.1切换模型(状态)
        model.train()   #训练模式
        # 2.3.2模型预测
        y_pred = model(x)
        # 2.3.3计算损失值
        loss = criterion(y_pred, y)
        # 2.3.4梯度清零，反向传播，优化参数
        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()
        # 2.3.5把每批次训练的平均损失值加起来
        total_loss += loss.item()
        batch_num += 1
    # 打印此轮训练信息
    print(f'第{epoch+1}轮的，loss=:{(total_loss/batch_num):.4f}, 耗时:{time.time()-start:.2f}秒')
    print('-'*60)
# 3保存模型参数
# 参1：模型对象的参数（权重矩阵，偏置矩阵），参2：模型保存的文件名
torch.save(model.state_dict(), "../output/model/arc.pth")#后缀名用：pth,pkl,pickle均可
# 参1：模型对象的参数（标准化参数），参2：模型保存的文件名
joblib.dump(scaler, "../output/model/arc_scaler.pkl")

# todo 4.模型测试
class_names = ['停机', '正常', '标准故障', '复合故障']
# 1.创建神经网络分类对象
model = DNN(input_dim, output_dim).to(device)
# 2.加载模型参数
model.load_state_dict(torch.load('../output/model/arc.pth'))
model.eval()    # 切换模型状态 -->测试模式
# 3.创建测试集的数据加载器对象
test_loader = DataLoader(test_dataset, batch_size=8,shuffle=False)
# 4.定义变量，记录测试的预测正确的样本数，混淆矩阵（行=真实类别，列=预测类别）
correct = 0
confusion = np.zeros((output_dim, output_dim), dtype=int)
# 5.从数据加载器中，获取到每批次的数据
with torch.no_grad():#     测试阶段不需要计算梯度，省显存、加速
    for x, y in test_loader:
        # 5.1把数据迁移到GPU
        x = x.to(device)
        y = y.to(device)
        # 5.2模型预测
        logits = model(x)
        # 5.3根据加权求和得到类别，用argmax(),获取最大值对应的下标，就是类别
        y_pred = torch.argmax(logits, dim=1)    #dim=1表示逐行处理
        # 5.4统计预测正确的样本个数
        correct += (y_pred == y).sum()
        # 5.5把真实标签和预测标签都转回CPU并统计到混淆矩阵
        y_true_np = y.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
        for t, p in zip(y_true_np, y_pred_np):
            confusion[t][p] += 1
# 6.走到这里，模型预测结束，打印准确率。
print(f'accuracy={100*correct/len(test_dataset):.2f}%')
print('-' * 70)
# 7.每个类型的准确率（召回率）：该类的真实样本中被判对的比例
print('每个类型的准确率：')
for i in range(output_dim):
    total_i = int(confusion[i, :].sum())
    correct_i = int(confusion[i, i])

    if total_i > 0:
        recall_i = 100.0 * correct_i / total_i
        recall_text = f'{recall_i:.2f}%'
    else:
        recall_text = '无测试样本'

    print(
        f'  类别 {class_names[i]:<8} '
        f'样本数:{total_i:>4}  '
        f'判对:{correct_i:>4}  '
        f'召回率:{recall_text}'
    )
print('-' * 70)
# 8.混淆矩阵：每一行 = 真实类别，每一列 = 被误判成的类别
print('混淆矩阵（行=真实类别，列=预测类别）：')
header = '真实\\预测 |' + ''.join(f'{class_names[j]:>8}' for j in range(output_dim))
print(header)
print('-' * len(header))
for i in range(output_dim):
    row = ''.join(f'{confusion[i][j]:>8}' for j in range(output_dim))
    print(f'{class_names[i]:>8} |{row}   <-- 共{confusion[i].sum()}个')

# 9.重点看：每个类型有多少被误判为其他类型
print('-' * 70)
print('各类型被误判为其他类型的情况：')
for i in range(output_dim):
    errors = {class_names[j]: int(confusion[i][j])
              for j in range(output_dim) if j != i and confusion[i][j] > 0}
    if errors:
        print(f'  {class_names[i]} 被误判为: {errors}')
    else:
        print(f'  {class_names[i]} 没有被误判，全部判对 ✔')




