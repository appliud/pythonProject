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

lr = 0.1  # 学习率为0.1
epochs = 500  # 循环轮数


# 生成新的神经网络参数
w1_sgd = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1_sgd = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

def sgd_optimizer(w, b, grads, lr):
    w.assign_sub(lr * grads[0])
    b.assign_sub(lr * grads[1])

# 训练SGD优化器
print("Training with SGD...")
start_time = time.time()

losses_sgd = []
accuracies_sgd = []

for epoch in range(epochs):
    epoch_loss = 0
    correct_predictions = 0

    for step, (x_batch_train, y_batch_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            y = tf.matmul(x_batch_train, w1_sgd) + b1_sgd
            y = tf.nn.softmax(y)
            y_ = tf.one_hot(y_batch_train, depth=3)
            loss = tf.reduce_mean(tf.square(y_ - y))

        grads = tape.gradient(loss, [w1_sgd, b1_sgd])
        epoch_loss += float(loss)

        predictions = tf.argmax(y, axis=1)
        predictions = tf.cast(predictions, dtype=y_batch_train.dtype)
        correct_predictions += tf.reduce_sum(tf.cast(tf.equal(predictions, y_batch_train), tf.float32))

        sgd_optimizer(w1_sgd, b1_sgd, grads, lr)

    avg_epoch_loss = epoch_loss / (step + 1)
    accuracy = correct_predictions / len(y_train)

    losses_sgd.append(avg_epoch_loss)
    accuracies_sgd.append(float(accuracy))

    if epoch % 50 == 0:
        print(f'SGD Epoch {epoch}, Loss: {avg_epoch_loss:.4f}, Accuracy: {float(accuracy):.4f}')

end_time = time.time()
training_time_sgd = end_time - start_time
print(f"SGD Training completed in {training_time_sgd:.2f} seconds.")
current_time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())

# 将training_time_sgd保存到txt文件里面，换行写入
with open('./img/time.txt', 'a') as f:
    f.write(f"\n{current_time} SGD Training completed in {training_time_sgd:.2f} seconds.")

# 绘制损失和准确率曲线
plt.figure(figsize=(10, 5))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(losses_sgd, label='SGD Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(accuracies_sgd, label='SGD Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()

# 保存图像
plt.savefig('./img/sgd_results_'+current_time+'.png')

plt.show()