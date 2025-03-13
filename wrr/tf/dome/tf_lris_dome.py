import matplotlib
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


iris = load_iris()
x_data = iris.data.astype(np.float32)  
y_data = iris.target.astype(np.int32)  


np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)


x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]


train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1, dtype=tf.float32))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1, dtype=tf.float32))


epochs = 500
lr = 0.1
test_acc = []
loss_list = []


for epoch in range(epochs):
    total_loss = 0
    for step, (x_batch, y_batch) in enumerate(train_db):
        with tf.GradientTape() as tape:
            
            y_pred = tf.matmul(x_batch, w1) + b1  
            
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=y_batch,
                    logits=y_pred
                )
            )
        
        grads = tape.gradient(loss, [w1, b1])
        
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        total_loss += loss.numpy()

    
    loss_list.append(total_loss / 4)

    
    total_correct = 0
    total_number = 0
    for x_test_batch, y_test_batch in test_db:
        y = tf.matmul(x_test_batch, w1) + b1
        y_prob = tf.nn.softmax(y)
        pred = tf.argmax(y_prob, axis=1)
        pred = tf.cast(pred, dtype=y_test_batch.dtype)
        correct = tf.cast(tf.equal(pred, y_test_batch), dtype=tf.int32)
        total_correct += tf.reduce_sum(correct).numpy()
        total_number += x_test_batch.shape[0]
    acc = total_correct / total_number
    test_acc.append(acc)

    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss_list[-1]:.4f}, Acc: {acc:.4f}")


plt.title("Acc Curve")
plt.xlabel("epoch")
plt.ylabel("acc")
plt.plot(test_acc, label="$Accuracy$")
plt.legend()
plt.show()