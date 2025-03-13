import tensorflow as tf

# 创建一个包含 0 到 9 的浮点数张量
i_values = tf.range(10, dtype=tf.float32)

# 计算这些值的 sigmoid
y_values = tf.nn.sigmoid(i_values)

# 打印结果
for y in y_values:
    print(y.numpy())
