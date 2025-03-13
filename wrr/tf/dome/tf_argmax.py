import tensorflow as tf
# 假设 x 是一个 (3, 3) 的二维张量
x = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 修改：将 axis 参数设置为 1，表示沿列方向计算最大值的索引
print(tf.argmax(x, axis=1))
