import tensorflow as tf

# 初始化特征和标签张量
features = tf.constant([12, 23, 10, 17])
labels = tf.constant([0, 1, 1, 0])

# 创建数据集
dataset = tf.data.Dataset.from_tensor_slices((features, labels))

# 打印数据集信息
print(dataset)

# 遍历数据集并打印每个元素
for element in dataset:
    print(element)

# 四则运算示例
x1 = tf.constant([1., 2., 3.], dtype=tf.float32)
x = tf.constant([[1., 2., 3.], [4., 5., 6.]], dtype=tf.float32)  # 修改 x 的数据类型为 float32

# 加法
add_result = tf.add(x, x1)
print("Addition Result:\n", add_result)

# 减法
sub_result = tf.subtract(x, x1)
print("Subtraction Result:\n", sub_result)

# 乘法
mul_result = tf.multiply(x, x1)
print("Multiplication Result:\n", mul_result)

# 除法
div_result = tf.divide(x, x1)
print("Division Result:\n", div_result)