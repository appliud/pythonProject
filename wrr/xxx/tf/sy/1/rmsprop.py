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
w1_rmsprop = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1_rmsprop = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

lr = 0.1  # 学习率为0.1
epochs = 500  # 循环轮数

def rmsprop_optimizer(w, b, grads, lr, v_w, v_b, beta=0.9):
    v_w = beta * v_w + (1 - beta) * tf.square(grads[0])
    v_b = beta * v_b + (1 - beta) * tf.square(grads[1])
    w.assign_sub(lr / tf.sqrt(v_w + 1e-7) * grads[0])
    b.assign_sub(lr / tf.sqrt(v_b + 1e-7) * grads[1])
    return v_w, v_b

# 训练RMSProp优化器
print("Training with RMSProp...")
start_time = time.time()

v_w, v_b = 0, 0
losses_rmsprop = []
accuracies_rmsprop = []

for epoch in range(epochs):
    epoch_loss = 0
    correct_predictions = 0

    for step, (x_batch_train, y_batch_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            y = tf.matmul(x_batch_train, w1_rmsprop) + b1_rmsprop
            y = tf.nn.softmax(y)
            y_ = tf.one_hot(y_batch_train, depth=3)
            loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.math.log(tf.clip_by_value(y, 1e-10, 1.0)), axis=1))  # 使用交叉熵损失

        grads = tape.gradient(loss, [w1_rmsprop, b1_rmsprop])
        batch_loss = float(loss)
        epoch_loss += batch_loss

        predictions = tf.argmax(y, axis=1)
        predictions = tf.cast(predictions, dtype=y_batch_train.dtype)
        correct_predictions += tf.reduce_sum(tf.cast(tf.equal(predictions, y_batch_train), tf.float32))

        v_w, v_b = rmsprop_optimizer(w1_rmsprop, b1_rmsprop, grads, lr, v_w, v_b)

    avg_epoch_loss = epoch_loss / (step + 1)
    accuracy = correct_predictions / len(y_train)

    losses_rmsprop.append(avg_epoch_loss)
    accuracies_rmsprop.append(float(accuracy))

    if epoch % 50 == 0:
        print(f'RMSProp Epoch {epoch}, Loss: {avg_epoch_loss:.4f}, Accuracy: {float(accuracy):.4f}')

end_time = time.time()
training_time_rmsprop = end_time - start_time
print(f"RMSProp Training completed in {training_time_rmsprop:.2f} seconds.")
current_time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())

# 将training_time_rmsprop保存到txt文件里面，换行写入
with open('./img/time.txt', 'a') as f:
    f.write(f"\n{current_time} RMSProp Training completed in {training_time_rmsprop:.2f} seconds.")

# 绘制损失和准确率曲线
plt.figure(figsize=(10, 5))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(losses_rmsprop, label='RMSProp Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(accuracies_rmsprop, label='RMSProp Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()

# 保存图像
plt.savefig('./img/rmsprop_results_' + current_time + '.png')

plt.show()