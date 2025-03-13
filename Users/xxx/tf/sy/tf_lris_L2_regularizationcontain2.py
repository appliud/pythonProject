import time
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')  # 或者 'Qt5Agg'

# 定义学习率调度器
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.1,  # 初始学习率
    decay_steps=1000,  # 衰减步长
    decay_rate=0.96,  # 衰减率
    staircase=True)  # 是否阶梯式下降

# 读入数据/标签 生成x_train y_train
filepath = "E:/class_2/dot.csv"  # 确保路径正确
df = pd.read_csv(filepath, header=0, encoding="gbk")

x_data = np.array(df[['x1', 'x2']])
y_data = np.array(df['y_c'])

x_train = x_data
y_train = y_data.reshape(-1, 1)

Y_c = [['red' if y else 'blue'] for y in y_train]

# 转换x的数据类型
x_train = tf.cast(x_train, tf.float32)
y_train = tf.cast(y_train, tf.float32)

# 创建Dataset对象
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

# 生成神经网络的参数
w1 = tf.Variable(tf.random.normal([2, 11]), dtype=tf.float32)
b1 = tf.Variable(tf.constant(0.01, shape=[11]))
w2 = tf.Variable(tf.random.normal([11, 1]), dtype=tf.float32)
b2 = tf.Variable(tf.constant(0.01, shape=[1]))

epochs = 500  # 循环轮数

# 记录开始时间
start_time = time.time()

# 初始化记录变量
losses = []
accuracies = []

# 训练部分
for epoch in range(epochs):
    current_lr = lr_schedule(tf.cast(epoch, tf.float32))  # 根据当前epoch计算学习率
    epoch_loss = 0
    correct_predictions = 0

    for step, (x_batch_train, y_batch_train) in enumerate(train_db):
        with tf.GradientTape() as tape:  # 记录梯度信息
            h1 = tf.matmul(x_batch_train, w1) + b1  # 记录神经网络乘加运算
            h1 = tf.nn.relu(h1)
            y_pred = tf.matmul(h1, w2) + b2

            # 采用均方误差损失函数mse = mean(sum(y-out)^2)
            loss_mse = tf.reduce_mean(tf.square(y_batch_train - y_pred))
            # 添加l2正则化
            loss_regularization = tf.reduce_sum([tf.nn.l2_loss(w1), tf.nn.l2_loss(w2)])
            loss = loss_mse + 0.03 * loss_regularization  # REGULARIZER = 0.03

        # 计算loss对各个参数的梯度
        variables = [w1, b1, w2, b2]
        grads = tape.gradient(loss, variables)

        # 实现梯度更新
        w1.assign_sub(current_lr * grads[0])
        b1.assign_sub(current_lr * grads[1])
        w2.assign_sub(current_lr * grads[2])
        b2.assign_sub(current_lr * grads[3])

        epoch_loss += float(loss)

        # 计算准确率
        predictions = tf.round(tf.sigmoid(y_pred))
        correct_predictions += tf.reduce_sum(tf.cast(tf.equal(predictions, y_batch_train), tf.float32))

    avg_epoch_loss = epoch_loss / (step + 1)
    accuracy = correct_predictions / len(y_train)

    losses.append(avg_epoch_loss)
    accuracies.append(float(accuracy))

    # 每50个epoch，打印loss信息
    if epoch % 50 == 0:
        print(
            f'Epoch {epoch}, Loss: {avg_epoch_loss:.4f}, Accuracy: {float(accuracy):.4f}, Learning Rate: {current_lr.numpy():.6f}')

end_time = time.time()
training_time = end_time - start_time
print(f"训练完成，耗时 {training_time:.2f} 秒。")

# 预测部分
print("*******predict*******")
# xx在-3到3之间以步长为0.01，yy在-3到3之间以步长0.01,生成间隔数值点
xx, yy = np.mgrid[-3:3:.1, -3:3:.1]
# 将xx, yy拉直，并合并配对为二维张量，生成二维坐标点
grid = np.c_[xx.ravel(), yy.ravel()]
grid = tf.cast(grid, tf.float32)
# 将网格坐标点喂入神经网络，进行预测，probs为输出
probs = []
for x_predict in grid:
    # 使用训练好的参数进行预测
    h1 = tf.matmul([x_predict], w1) + b1
    h1 = tf.nn.relu(h1)
    y = tf.matmul(h1, w2) + b2  # y为预测结果
    probs.append(y)

# 取第0列给x1，取第1列给x2
x1 = x_data[:, 0]
x2 = x_data[:, 1]
# probs的shape调整成xx的样子
probs = np.array(probs).reshape(xx.shape)

plt.figure(figsize=(12, 4))

# Loss图
plt.subplot(1, 3, 1)
plt.plot(range(epochs), losses, label='Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# ACC图
plt.subplot(1, 3, 2)
plt.plot(range(epochs), accuracies, label='Accuracy', color='green')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 训练耗时对比图（这里仅展示本次训练的耗时）
plt.subplot(1, 3, 3)
plt.bar(['Training Time'], [training_time], color='skyblue')
plt.title('Training Time')
plt.ylabel('Time (seconds)')

plt.tight_layout()
plt.savefig('training_summary.png')  # 保存图片
plt.show()