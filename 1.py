import tensorflow as tf

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0: DEBUG, 1: INFO, 2: WARNING, 3: ERROR

# 初始化一个常量张量a，包含两个浮点数
a = tf.constant([1.0, 2.0], name="a")
# 输出张量a的信息
print("a:",a)
# 输出a的值
print("a.numpy()值:",a.numpy())
# 输出张量a的数据类型
print("a.dtype类型:",a.dtype)
# 输出张量a的形状
print(a.shape, "\n")

# 初始化一个正态分布随机张量b，均值为0.5，标准差为1
b = tf.random.normal([2, 2], mean=0.5, stddev=1)
# 输出张量b的信息，其值分布较为广泛
print(b)  # 分布的更多
# 初始化一个截断正态分布随机张量c，均值为0.5，标准差为1
# 截断正态分布意味着只保留均值附近一定范围内的值
c = tf.random.truncated_normal([2, 2], mean=0.5, stddev=1)
# 输出张量c的信息
print("\nc:\n",c)

# 初始化一个均匀分布随机张量d，值域在0到1之间
d = tf.random.uniform([2, 2], minval=0, maxval=1)
# 输出张量d的信息
print("\nd_min0_max1:\n",d)

x1 = tf.constant([1., 2., 3.], dtype=tf.float32)
print(x1)
x = tf.constant([[1., 2., 3.], [4., 5., 6.]])
print("\n x",x)
print("x_ave",tf.reduce_mean(x).numpy())  # 求平均值
print("x_add",tf.reduce_sum(x).numpy())  # 求和

