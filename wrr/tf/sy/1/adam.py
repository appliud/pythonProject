import os
import time
import matplotlib
import tensorflow as tf
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

# 导入数据，分别为输入特征和标签
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

# 随机打乱数据
np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

# 将打乱后的数据集分割为训练集和测试集
x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

# 转换x的数据类型
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

# 创建Dataset对象
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 生成神经网络的参数
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

lr = 0.1  # 学习率调整为适合Adam优化器的值
epochs = 500  # 循环轮数

# 使用Adam优化器
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

train_loss_results = []  # 将每轮的loss记录在此列表中，为后续画loss曲线提供数据
test_acc = []  # 将每轮的acc记录在此列表中，为后续画acc曲线提供数据

# 训练部分
now_time = time.time()
for epoch in range(epochs):  # 数据集级别的循环，每个epoch循环一次数据集
    epoch_loss_avg = tf.keras.metrics.Mean()  # 每个epoch的平均损失
    correct_predictions = 0  # 正确预测的数量

    for step, (x_batch_train, y_batch_train) in enumerate(train_db):  # batch级别的循环 ，每个step循环一个batch
        with tf.GradientTape() as tape:  # with结构记录梯度信息
            y = tf.matmul(x_batch_train, w1) + b1  # 神经网络乘加运算
            y = tf.nn.softmax(y)  # 使输出y符合概率分布（此操作后与独热码同量级，可相减求loss）
            y_ = tf.one_hot(y_batch_train, depth=3)  # 将标签值转换为独热码格式，方便计算loss和accuracy
            loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.math.log(tf.clip_by_value(y, 1e-10, 1.0)), axis=1))  # 使用交叉熵损失

        grads = tape.gradient(loss, [w1, b1])
        optimizer.apply_gradients(zip(grads, [w1, b1]))  # 应用梯度

        # 更新损失和准确率统计
        epoch_loss_avg.update_state(loss)
        predictions = tf.argmax(y, axis=1)
        predictions = tf.cast(predictions, dtype=y_batch_train.dtype)
        correct_predictions += tf.reduce_sum(tf.cast(tf.equal(predictions, y_batch_train), tf.float32))

    # 计算当前epoch的平均损失和准确率
    train_loss_results.append(epoch_loss_avg.result())
    accuracy = correct_predictions / len(y_train)
    test_acc.append(float(accuracy))

    if epoch % 50 == 0:
        print(f'Adam Epoch {epoch}, Loss: {epoch_loss_avg.result():.4f}, Accuracy: {float(accuracy):.4f}')

total_time = time.time() - now_time
print(f"Adam Training completed in {total_time:.2f} seconds.")
current_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())

# 将训练时间保存到txt文件里面
with open('img/time.txt', 'a') as f:
    f.write(f"\n{current_time} adma Training completed in {total_time:.2f} seconds.")

# 绘制损失和准确率曲线
plt.figure(figsize=(10, 5))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(train_loss_results, label='Adam Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(test_acc, label='Adam Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()

# 保存图像
plt.savefig('./img/adma_results_' + current_time + '.png')
plt.show()