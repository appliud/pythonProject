# 导入必要的库
import matplotlib
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris

# 设置matplotlib的后端为TkAgg以确保在大多数环境中都能正常显示图形界面
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 数据集读入与预处理
# 加载Iris数据集，这是一个经典的数据集，包含150个样本，每个样本有4个特征（花萼长度、花萼宽度、花瓣长度、花瓣宽度），共3类鸢尾花
iris = load_iris()
# 将数据转换为适合TensorFlow使用的数据类型：特征x_data为float32，标签y_data为int32
x_data, y_data = iris.data.astype(np.float32), iris.target.astype(np.int32)

# 设置随机种子并打乱数据，确保每次运行时数据顺序一致，以便结果可复现
np.random.seed(116)
shuffle_indices = np.random.permutation(len(x_data))
x_data, y_data = x_data[shuffle_indices], y_data[shuffle_indices]

# 划分训练集和测试集
# 通常我们会将大部分数据用于训练模型，少部分用于评估模型性能。这里我们使用最后30个样本作为测试集
split_point = -30
x_train, y_train = x_data[:split_point], y_data[:split_point]  # 前面的数据作为训练集
x_test, y_test = x_data[split_point:], y_data[split_point:]    # 最后的30个样本作为测试集

# 创建TensorFlow Dataset对象
# 使用.shuffle(100)来增加数据多样性，并使用.batch(32)设置批次大小为32，这样每次迭代只取32个样本进行训练
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(100).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 定义网络参数
# 初始化权重w1和偏置b1，定义一个简单的线性分类器。输入维度为4（对应于4个特征），输出维度为3（对应于3种不同的鸢尾花）
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1,seed=1 ,dtype=tf.float32))  # 权重矩阵
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1,seed=1 , dtype=tf.float32))     # 偏置向量

# 超参数设置
epochs = 500  # 训练轮数，即整个数据集被用来训练多少次
lr = 0.1      # 学习率，控制参数更新的速度

test_acc = [] # 用于存储每个epoch的测试准确率
loss_list = []# 用于存储每个epoch的平均损失值

# 训练循环
for epoch in range(epochs):
    total_loss = 0  # 每个epoch开始时重置总损失
    for step, (x_batch, y_batch) in enumerate(train_db):
        with tf.GradientTape() as tape:#求导数
            # 前向传播：计算预测值y_pred。使用tf.matmul进行矩阵乘法运算，得到每个样本属于各个类别的分数
            y_pred = tf.matmul(x_batch, w1) + b1
            # 计算损失：使用交叉熵损失函数计算当前批次的损失。sparse_softmax_cross_entropy_with_logits适用于类别标签是整数的情况
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_batch, logits=y_pred))
        # 反向传播及参数更新：计算梯度并更新权重和偏置
        grads = tape.gradient(loss, [w1, b1])  # 计算梯度
        w1.assign_sub(lr * grads[0])  # 更新权重
        b1.assign_sub(lr * grads[1])  # 更新偏置
        total_loss += loss.numpy()    # 累加批次损失

    avg_loss = total_loss / len(train_db)  # 计算本epoch的平均损失
    loss_list.append(avg_loss)             # 保存平均损失

    # 测试集评估
    total_correct, total_number = 0, 0
    for x_test_batch, y_test_batch in test_db:
        y = tf.matmul(x_test_batch, w1) + b1
        pred = tf.argmax(tf.nn.softmax(y), axis=1)  # 对每个样本计算其属于各分类的概率，并取概率最大的类别作为预测结果
        pred = tf.cast(pred, dtype=y_test_batch.dtype)  # 类型转换，确保预测结果与真实标签类型一致
        correct = tf.cast(tf.equal(pred, y_test_batch), dtype=tf.int32)  # 计算预测正确的样本数量
        total_correct += tf.reduce_sum(correct).numpy()  # 累加正确数量
        total_number += x_test_batch.shape[0]            # 累加总数
    acc = total_correct / total_number                   # 计算准确率
    test_acc.append(acc)                                 # 保存准确率

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Test Acc: {acc:.4f}")# 输出当前epoch的信息

# 可视化结果
plt.figure(figsize=(12, 5))  # 设置图像大小为宽12英寸，高5英寸
# 绘制损失曲线
plt.subplot(1, 2, 1)         # 在一个图中创建两个子图，这是第一个子图
plt.title("Loss Curve")       # 标题为“损失曲线”
plt.xlabel("epoch")           # X轴标签为“epoch”，表示迭代次数
plt.ylabel("loss")            # Y轴标签为“loss”，表示损失值
plt.plot(loss_list, label="$Loss$")  # 绘制损失变化曲线，并标注为“Loss”
plt.legend()                  # 显示图例

# 绘制准确率曲线
plt.subplot(1, 2, 2)         # 这是第二个子图
plt.title("Accuracy Curve")   # 标题为“准确率曲线”
plt.xlabel("epoch")           # X轴标签为“epoch”，表示迭代次数
plt.ylabel("accuracy")        # Y轴标签为“accuracy”，表示准确率
plt.plot(test_acc, label="$Test Accuracy$", color='r')  # 绘制准确率变化曲线，并标注为“Test Accuracy”，颜色设为红色
plt.legend()                  # 显示图例

plt.show()  # 显示绘制的图像
'''
这两张图表展示了机器学习模型在训练过程中的表现。以下是每张图表的详细解释：

### 左图：损失曲线
- **标题**：Loss Curve（损失曲线）
- **X轴**：Epoch（迭代次数，即完整数据集通过的次数）
- **Y轴**：Loss（损失值，衡量模型在训练数据上的表现）

**关键观察点**：
- 损失值在开始时较高，并随着迭代次数的增加而逐渐下降。
- 这表明模型正在学习并改进其性能。
- 损失值有一些波动，这是由于优化过程的随机性导致的。

### 右图：准确率曲线
- **标题**：Accuracy Curve（准确率曲线）
- **X轴**：Epoch（迭代次数，即完整数据集通过的次数）
- **Y轴**：Accuracy（准确率，衡量模型在测试数据上的分类效果）

**关键观察点**：
- 准确率在开始时较低，并随着迭代次数的增加而逐渐上升。
- 初始阶段准确率有较大的波动，表明模型还在学习和调整参数。
- 经过一定数量的迭代后，准确率趋于稳定，并达到接近1.0的高值，表明模型已经学会了很好地分类测试数据。

### 详细分析：
1. **损失曲线**：
   - **初始高损失**：刚开始训练时，模型尚未学会数据中的模式，因此损失值较高。
   - **损失下降**：随着模型训练，它不断调整参数以最小化损失，导致损失值逐渐下降。
   - **波动**：损失曲线中的小波动表明模型偶尔会犯错或遇到难以处理的例子。

2. **准确率曲线**：
   - **初始低准确率**：刚开始时，模型错误较多，因此准确率较低。
   - **准确率上升**：随着模型学习，它变得越来越好地分类数据，导致准确率逐渐上升。
   - **稳定**：经过一定时间后，准确率趋于稳定并保持在较高水平，表明模型已经收敛并且在测试数据上表现一致。

### 结论：
- 左图显示模型有效地减少了损失值，这是一个良好的学习迹象。
- 右图显示模型在测试集上达到了较高的准确率，表明模型对未见过的数据具有良好的泛化能力。
- 这两张图表结合在一起，全面展示了模型的训练进度和性能。

如果你有任何具体问题或需要进一步分析，请随时提问！
'''