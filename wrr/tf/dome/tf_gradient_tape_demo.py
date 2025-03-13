import tensorflow as tf

# 定义一个简单的函数y = x^2
def simple_function(x):
    return x ** 2

# 使用tf.GradientTape计算梯度
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    # 在tape上下文中进行的操作会被记录下来用于梯度计算
    y = simple_function(x)

# 计算y关于x的梯度
dy_dx = tape.gradient(y, x)

print(f"Gradient of y with respect to x at x=3 is: {dy_dx.numpy()}")

# 更复杂的例子：计算多个变量的梯度
w = tf.Variable(tf.random.normal((2, 2)), name='w')
b = tf.Variable(tf.zeros(2, dtype=tf.float32), name='b')
x = [[1., 2.], [3., 4.]]

with tf.GradientTape(persistent=True) as tape:
    y = x @ w + b
    loss = tf.reduce_mean(y**2)

# 分别计算loss对w和b的梯度
dw = tape.gradient(loss, w)
db = tape.gradient(loss, b)

print("Gradients:")
print(f"dL/dw:\n{dw}")
print(f"dL/db:\n{db}")

# 删除持久性带子以释放资源
del tape