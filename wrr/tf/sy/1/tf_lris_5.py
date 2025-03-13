import time
import tensorflow as tf
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

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


# 定义优化器函数
def sgd_optimizer(w, b, grads, lr):
    w.assign_sub(lr * grads[0])
    b.assign_sub(lr * grads[1])


def sgdm_optimizer(w, b, grads, lr, m_w, m_b, beta=0.9):
    m_w = beta * m_w + (1 - beta) * grads[0]
    m_b = beta * m_b + (1 - beta) * grads[1]
    w.assign_sub(lr * m_w)
    b.assign_sub(lr * m_b)
    return m_w, m_b


def adagrad_optimizer(w, b, grads, lr, g_w_sum, g_b_sum):
    g_w_sum += tf.square(grads[0])
    g_b_sum += tf.square(grads[1])
    w.assign_sub(lr / tf.sqrt(g_w_sum + 1e-7) * grads[0])
    b.assign_sub(lr / tf.sqrt(g_b_sum + 1e-7) * grads[1])
    return g_w_sum, g_b_sum


def rmsprop_optimizer(w, b, grads, lr, v_w, v_b, beta=0.9):
    v_w = beta * v_w + (1 - beta) * tf.square(grads[0])
    v_b = beta * v_b + (1 - beta) * tf.square(grads[1])
    w.assign_sub(lr / tf.sqrt(v_w + 1e-7) * grads[0])
    b.assign_sub(lr / tf.sqrt(v_b + 1e-7) * grads[1])
    return v_w, v_b


def adam_optimizer(w, b, grads, lr, m_w, m_b, v_w, v_b, global_step, beta1=0.9, beta2=0.999):
    m_w = beta1 * m_w + (1 - beta1) * grads[0]
    m_b = beta1 * m_b + (1 - beta1) * grads[1]
    v_w = beta2 * v_w + (1 - beta2) * tf.square(grads[0])
    v_b = beta2 * v_b + (1 - beta2) * tf.square(grads[1])
    m_w_correction = m_w / (1 - tf.pow(beta1, int(global_step)))
    m_b_correction = m_b / (1 - tf.pow(beta1, int(global_step)))
    v_w_correction = v_w / (1 - tf.pow(beta2, int(global_step)))
    v_b_correction = v_b / (1 - tf.pow(beta2, int(global_step)))
    w.assign_sub(lr * m_w_correction / tf.sqrt(v_w_correction + 1e-7))
    b.assign_sub(lr * m_b_correction / tf.sqrt(v_b_correction + 1e-7))
    return m_w, m_b, v_w, v_b


optimizers = {
    'sgd': sgd_optimizer,
    'sgdm': sgdm_optimizer,
    'adagrad': adagrad_optimizer,
    'rmsprop': rmsprop_optimizer,
    'adam': adam_optimizer
}

optimizer_results = {}

for opt_name, optimizer in optimizers.items():
    print(f"Training with {opt_name}...")

    # 初始化参数
    w1_opt = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
    b1_opt = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

    if opt_name == 'sgdm':
        m_w, m_b = 0, 0
    elif opt_name == 'adagrad':
        g_w_sum, g_b_sum = 0, 0
    elif opt_name == 'rmsprop':
        v_w, v_b = 0, 0
    elif opt_name == 'adam':
        m_w, m_b, v_w, v_b = 0, 0, 0, 0
        global_step = 0

    losses = []
    accuracies = []

    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0
        correct_predictions = 0

        for step, (x_batch_train, y_batch_train) in enumerate(train_db):
            with tf.GradientTape() as tape:
                y = tf.matmul(x_batch_train, w1_opt) + b1_opt
                y = tf.nn.softmax(y)
                y_ = tf.one_hot(y_batch_train, depth=3)
                loss = tf.reduce_mean(tf.square(y_ - y))

            grads = tape.gradient(loss, [w1_opt, b1_opt])
            epoch_loss += float(loss)

            predictions = tf.argmax(y, axis=1)
            predictions = tf.cast(predictions, dtype=y_batch_train.dtype)
            correct_predictions += tf.reduce_sum(tf.cast(tf.equal(predictions, y_batch_train), tf.float32))

            if opt_name == 'sgdm':
                m_w, m_b = optimizer(w1_opt, b1_opt, grads, lr, m_w, m_b)
            elif opt_name == 'adagrad':
                g_w_sum, g_b_sum = optimizer(w1_opt, b1_opt, grads, lr, g_w_sum, g_b_sum)
            elif opt_name == 'rmsprop':
                v_w, v_b = optimizer(w1_opt, b1_opt, grads, lr, v_w, v_b)
            elif opt_name == 'adam':
                m_w, m_b, v_w, v_b = optimizer(w1_opt, b1_opt, grads, lr, m_w, m_b, v_w, v_b, global_step)
                global_step += 1
            else:
                optimizer(w1_opt, b1_opt, grads, lr)

        avg_epoch_loss = epoch_loss / (step + 1)
        accuracy = correct_predictions / len(y_train)

        losses.append(avg_epoch_loss)
        accuracies.append(float(accuracy))

        if epoch % 50 == 0:
            print(f'{opt_name} Epoch {epoch}, Loss: {avg_epoch_loss:.4f}, Accuracy: {float(accuracy):.4f}')

    end_time = time.time()
    training_time = end_time - start_time

    optimizer_results[opt_name] = {
        'losses': losses,
        'accuracies': accuracies,
        'training_time': training_time
    }

# 绘制结果图表
plt.figure(figsize=(15, 10))

for i, (opt_name, result) in enumerate(optimizer_results.items(), 1):
    plt.subplot(3, 2, i)
    plt.plot(range(epochs), result['losses'], label='Loss')
    plt.title(f'{opt_name.upper()} Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

for i, (opt_name, result) in enumerate(optimizer_results.items(), 4):
    plt.subplot(3, 2, i)
    plt.plot(range(epochs), result['accuracies'], label='Accuracy', color='green')
    plt.title(f'{opt_name.upper()} Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

plt.subplot(3, 2, 6)
optimizers_names = list(optimizer_results.keys())
training_times = [result['training_time'] for result in optimizer_results.values()]
plt.bar(optimizers_names, training_times, color='skyblue')
plt.title('Training Time Comparison')
plt.ylabel('Time (seconds)')

plt.tight_layout()
plt.savefig('optimizers_comparison.png')  # 保存图片
plt.show()