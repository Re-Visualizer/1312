# todo 5.检测一个完整电弧的状态
import torch
import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from utils.nnclass import ArcClassificationModel

# 设置中文字体，解决图中中文/全角符号显示为方框(□)及 Glyph missing 警告
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 统一定义训练设备：有GPU用GPU，否则用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'当前使用的训练设备：{device}')

# 1.读取一个完整电弧周期的CSV数据
new_data = pd.read_csv("../data/58复合3-2.csv")

# 2.取出特征数据，最后一列是状态标签
feature, label_new= new_data.iloc[:, 0:-1], new_data.iloc[:,-1]

# 3.把特征列转为浮点型，标签转化为整型
feature = feature.astype(np.float32)
label_new = label_new.astype(np.int64)

# 4.加载标准化器,标准化原始数据
scaler = joblib.load("../output/model/arc_scaler.pkl")
feature = scaler.transform(feature).astype(np.float32)
feature = torch.from_numpy(feature).to(device)

# 5.创建神经网络分类对象，加载训练好的模型和标准化器
model = ArcClassificationModel(2, 4).to(device)
model.load_state_dict(torch.load('../output/model/arc.pth', map_location=device))

# 6.模型状态(切换为测试状态)
model.eval()

# 7.预测完整电弧每一行数据的状态
with torch.no_grad():
    y_pred = model(feature)
    y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()
# 8.多周期判断：10个周期中超过7个故障状态，则判定发生故障
# 类别2是单故障，类别3是复合故障
fault = np.isin(y_pred, [2, 3]).astype(np.int64)
fault_count = pd.Series(fault).rolling(window=10).sum()
fault_index = np.where(fault_count > 7)[0]
# 一旦判定发生故障，此后均为故障状态
fault_result = np.zeros(len(y_pred), dtype=np.int64)
if len(fault_index) > 0:
    fault_result[fault_index[0]:] = 1

# 9.保存完整电弧的状态预测结果
result = new_data.copy()
result["预测类别"] = y_pred
result["故障判断"] = fault_result
print(result)
result.to_csv("../output/预测结果.csv", index=False, encoding="utf-8-sig")

# 10.生成特征与故障判断结果对应图
x = np.arange(len(new_data))
# 把真实标签转换成 0/1 故障真值（类别2、3为故障），用于和故障判断结果对比
label_true = np.isin(label_new.values, [2, 3]).astype(np.int64)
# 类别编号 -> 中文名称
class_names = {0: "停机", 1: "正常", 2: "标准故障", 3: "复合故障"}
fig, ax1 = plt.subplots(figsize=(12, 6))
# 左侧y轴：特征曲线
ax1.plot(x, new_data.iloc[:, 0], color="blue", label="feature")
ax1.set_xlabel("cycle")
ax1.set_ylabel("feature")
# 右侧y轴：真实故障真值 vs 故障判断结果
ax2 = ax1.twinx()
ax2.step(x, label_true,   where="post", color="green", linestyle="--", label="true fault")
ax2.step(x, fault_result, where="post", color="red",  label="fault result")
ax2.set_ylabel("fault state (0/1)")
ax2.set_yticks([0, 1])
# 在 fault_result 上标注首次判定故障的时刻及该时刻的类别判断
if len(fault_index) > 0:
    first_pos = int(fault_index[0])              # 首次判定故障的位置
    first_class = int(y_pred[first_pos])          # 该时刻模型的类别判断
    ax2.annotate(
        f"首次判定故障（第{first_pos}周期）\n"
        f"类别：{first_class} - {class_names[first_class]}",
        xy=(first_pos, 1),
        xytext=(first_pos + len(x) * 0.03, 0.72),   # 文字放在触发点右下方，避免压线
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        fontsize=11,
        color="black",
        bbox=dict(boxstyle="round,pad=0.4", fc="yellow", alpha=0.6, edgecolor="black"),
    )
    ax2.plot(first_pos, 1, marker="o", color="black", markersize=8, zorder=5)  # 触发点标记

ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.tight_layout()
# 同时保存 PNG 和 SVG（SVG 用于后续导入 Visio，见第3点说明）
plt.savefig("../output/特征与故障判断结果.png", dpi=300)
plt.savefig("../output/特征与故障判断结果.svg")
plt.show()