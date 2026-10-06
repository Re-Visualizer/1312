import time
import torch
import joblib
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset import create_CNNdataset
from utils.ArcClassificationModel import CNN

# configure
torch.manual_seed(23)                   # 随机种子

# 数据集参数
CSV_PATH = '../data/CNN4分类.csv'        # 数据集路径
class_names = ['停机', '正常', '标准故障', '复合故障']
TSET_SIZE = 0.3                         # 测试集占比
RANDOM_STATE = 23                       # 随机种子

# Adam优化器参数
LR = 1e-3                               # 学习率
BETAS = (0.9, 0.999)                    # 动量因子
WEIGHT_DECAY = 1e-4                    # 权重衰减

# 训练参数
EPOCHS = 150                            # 训练轮数
BATCH_SIZE = 32                         # 每批次样本数

# 模型保存
model_path = '../output/model/arc_model.pth'
scaler_path = '../output/model/arc_scaler.pkl'

# 统一定义训练设备：有GPU用GPU，否则用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'当前使用的训练设备：{device}')

# todo 1.构建数据集
train_dataset, test_dataset, output_dim, scaler= create_CNNdataset(CSV_PATH, TSET_SIZE, RANDOM_STATE)
# 1.创建数据加载器
dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,shuffle=True)

# todo 2.搭建神经网络
# 1.使用神经网络ArcModel,创建对象model
model = CNN().to(device)
# 2.定义损失函数，因为是多分类，这里用的是：多分类交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 3.创建优化器对象
optimizer = optim.AdamW(model.parameters(),lr=LR,betas=BETAS,weight_decay=WEIGHT_DECAY)

# todo 3.模型训练
# 1.遍历，完成每轮的 所有批次的训练动作
for epoch in range(EPOCHS):
    # 1.1定义变量， 记录总损失，总样本数据量，预测正确样本个数，训练(开始)时间
    total_loss, total_samples, total_correct = 0.0, 0, 0
    # 1.2定义变量，表示训练开始的时间
    start = time.time()
    # 1.3遍历数据加载器，获取 每批次的 数据
    for x,y in dataloader:
        # 1.3.0把数据迁移到GPU
        x = x.to(device)
        y = y.to(device)
        # 1.3.1切换训练模式
        model.train()
        # 1.3.2模型预测
        y_pred = model(x)
        # 1.3.3计算损失
        loss = criterion(y_pred, y)
        # 1.3.4 梯度清零 + 反向传播 + 参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 1.3.5统计总损失
        total_loss += loss.item()*len(y)
        # 1.3.6统计预测正确的样本个数
        total_correct += (torch.argmax(y_pred,dim=-1)==y).sum()
        # 1.3.7统计总样本个数
        total_samples += len(y)
    # 1.4走这里，说明一轮的训练结束，打印该轮的训练信息
    print(f"第:{epoch+1}轮的: loss={total_loss/total_samples:.5f},accuracy={total_correct/total_samples:.2%}, 耗时:{time.time()-start:.2f}秒")
    print('-'*60)

# todo 4.保存模型
# 参1：模型对象的参数（权重矩阵，偏置矩阵），参2：模型保存的文件名
torch.save(model.state_dict(), model_path)
joblib.dump(scaler, scaler_path)

# todo 5.模型测试
# 1.创建神经网络分类对象
model = CNN().to(device)
# 2.加载模型参数
model.load_state_dict(torch.load(model_path))
model.eval()    # 切换模型状态 -->测试模式
# 3.创建测试集的数据加载器对象
test_loader = DataLoader(test_dataset, batch_size=16,shuffle=False)
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
        y_pred = torch.argmax(logits, dim=1)  # dim=1表示逐行处理
        # 5.4统计预测正确的样本个数
        correct += (y_pred == y).sum()
        # 5.5把真实标签和预测标签都转回CPU并统计到混淆矩阵
        y_true_np = y.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
        for t, p in zip(y_true_np, y_pred_np):
            confusion[t][p] += 1
# 6.走到这里，模型预测结束，打印准确率。
print(f'accuracy={100 * correct / len(test_dataset):.2f}%')
print('-' * 70)
# 7.每个类型的准确率（召回率）：该类的真实样本中被判对的比例
print('每个类型的准确率：')
for i in range(output_dim):
    total_i = int(confusion[i, :].sum())
    correct_i = int(confusion[i, i])
    recall_text = f'{100.0 * correct_i / total_i:.2f}%'
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

