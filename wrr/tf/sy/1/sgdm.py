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
w1_sgdm = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1_sgdm = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

lr = 0.1  # 学习率为0.1
epochs = 500  # 循环轮数


# 生成新的神经网络参数
w1_sgdm = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1_sgdm = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

def sgdm_optimizer(w, b, grads, lr, m_w, m_b, beta=0.9):
    m_w = beta * m_w + (1 - beta) * grads[0]
    m_b = beta * m_b + (1 - beta) * grads[1]
    w.assign_sub(lr * m_w)
    b.assign_sub(lr * m_b)
    return m_w, m_b

# 训练SGDM优化器
print("Training with SGDM...")
start_time = time.time()

m_w, m_b = 0, 0
losses_sgdm = []
accuracies_sgdm = []

for epoch in range(epochs):
    epoch_loss = 0
    correct_predictions = 0

    for step, (x_batch_train, y_batch_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            y = tf.matmul(x_batch_train, w1_sgdm) + b1_sgdm
            y = tf.nn.softmax(y)
            y_ = tf.one_hot(y_batch_train, depth=3)
            loss = tf.reduce_mean(tf.square(y_ - y))

        grads = tape.gradient(loss, [w1_sgdm, b1_sgdm])
        epoch_loss += float(loss)

        predictions = tf.argmax(y, axis=1)
        predictions = tf.cast(predictions, dtype=y_batch_train.dtype)
        correct_predictions += tf.reduce_sum(tf.cast(tf.equal(predictions, y_batch_train), tf.float32))

        m_w, m_b = sgdm_optimizer(w1_sgdm, b1_sgdm, grads, lr, m_w, m_b)

    avg_epoch_loss = epoch_loss / (step + 1)
    accuracy = correct_predictions / len(y_train)

    losses_sgdm.append(avg_epoch_loss)
    accuracies_sgdm.append(float(accuracy))

    if epoch % 50 == 0:
        print(f'SGDM Epoch {epoch}, Loss: {avg_epoch_loss:.4f}, Accuracy: {float(accuracy):.4f}')

end_time = time.time()
training_time_sgdm = end_time - start_time
print(f"SGDM Training completed in {training_time_sgdm:.2f} seconds.")
current_time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())

# 将training_time_sgdm保存到txt文件里面，换行写入
with open('img/time.txt', 'a') as f:
    f.write(f"\n{current_time} SGDM Training completed in {training_time_sgdm:.2f} seconds.")

# 绘制损失和准确率曲线
plt.figure(figsize=(10, 5))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(losses_sgdm, label='SGDM Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(accuracies_sgdm, label='SGDM Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()

# 保存图像
plt.savefig('./img/sgdm_results_'+current_time+'.png')

plt.show()