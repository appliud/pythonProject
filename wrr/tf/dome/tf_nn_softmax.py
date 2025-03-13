# tf.nn.softmax() 使输出符合概率分布
# nn神经网络
import tensorflow as tf

x = tf.constant([[1., 2., 3.], [4., 5., 6.]])
print(x)
print(tf.nn.softmax(x))