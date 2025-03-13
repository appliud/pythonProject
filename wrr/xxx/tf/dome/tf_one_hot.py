# 独热编码
import tensorflow as tf


def one_hot_encode(labels, num_classes):
    """
    对标签进行独热编码

    参数：
    labels：标签列表，每个标签是一个整数
    num_classes：标签的类别数 几列

    返回值：
    one_hot_labels：独热编码后的标签列表，每个标签是一个one-hot向量
    """

    # 将标签转换为one-hot向量
    one_hot_labels = tf.one_hot(labels, depth=num_classes)

    return one_hot_labels


# 示例用法1
labels = [0, 1, 2, 3]
num_classes = 4
one_hot_labels = one_hot_encode(labels, num_classes)
print(one_hot_labels)

# 示例用法2
labels = [3, 2, 1, 0]
num_classes = 4
one_hot_labels = one_hot_encode(labels, num_classes)
print(one_hot_labels)
