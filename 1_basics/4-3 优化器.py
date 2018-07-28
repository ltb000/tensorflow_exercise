
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# In[18]:


# 载入数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 每个批次的大小
batch_size = 50
# 计算一共有多少个批次
n_batch = mnist.train.num_examples//batch_size

# 定义占位符
x = tf.placeholder(tf.float32, [None,784])
y = tf.placeholder(tf.float32, [None,10])
keep_prob = tf.placeholder(tf.float32)
lr = tf.Variable(0.001,tf.float32)

# 创建一个简单的神经网络
# W = tf.Variable(tf.zeros([784,10]))
# b = tf.Variable(tf.zeros([10]))
# 
W1 = tf.Variable(tf.truncated_normal([784,500], stddev=0.1))
b1 = tf.Variable(tf.zeros([500])+0.1)
L1 = tf.nn.tanh(tf.matmul(x,W1)+b1)
# L1 = tf.nn.relu(tf.matmul(x,W1)+b1)
L1_drop = tf.nn.dropout(L1,keep_prob)

W2 = tf.Variable(tf.truncated_normal([500,100], stddev=0.1))
b2 = tf.Variable(tf.zeros([100])+0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop,W2)+b2)
# L2 = tf.nn.relu(tf.matmul(L1_drop,W2)+b2)
L2_drop = tf.nn.dropout(L2,keep_prob)

W3 = tf.Variable(tf.truncated_normal([100,10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10])+0.1)
#
prediction = tf.nn.softmax(tf.matmul(L2_drop,W3)+b3)

# 二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
# 交叉熵
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# loss = -tf.reduce_sum(y*tf.log(prediction))
# 使用梯度下降法
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
# 优化器
train_step = tf.train.AdamOptimizer(lr).minimize(loss)
# 初始化变量
init = tf.global_variables_initializer()

# 预测准确率 
# 先生成一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(prediction,1))
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(51):
        learning_rate = sess.run(lr.assign(0.001*(0.9**epoch)))
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys, keep_prob:1.0})
            
        test_acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
        train_acc = sess.run(accuracy, feed_dict={x:mnist.train.images, y:mnist.train.labels, keep_prob:1.0})
        print('Iter',epoch,',Testing Accuracy',test_acc,',Training Accuracy',train_acc,',Learning rate',learning_rate)
        


# In[ ]:


Iter 49 ,Testing Accuracy 0.9777 ,Training Accuracy 0.99161816
Iter 50 ,Testing Accuracy 0.979 ,Training Accuracy 0.9918727

Iter 0 ,Testing Accuracy 0.9415 ,Training Accuracy 0.9430364
Iter 1 ,Testing Accuracy 0.9552 ,Training Accuracy 0.95870906
Iter 2 ,Testing Accuracy 0.9646 ,Training Accuracy 0.9684182
Iter 3 ,Testing Accuracy 0.9688 ,Training Accuracy 0.9750364
Iter 4 ,Testing Accuracy 0.9683 ,Training Accuracy 0.97743636
Iter 5 ,Testing Accuracy 0.9727 ,Training Accuracy 0.98185456
Iter 6 ,Testing Accuracy 0.9733 ,Training Accuracy 0.9845273
Iter 7 ,Testing Accuracy 0.9768 ,Training Accuracy 0.98685455
Iter 8 ,Testing Accuracy 0.9774 ,Training Accuracy 0.98825455
Iter 9 ,Testing Accuracy 0.977 ,Training Accuracy 0.9890364
Iter 10 ,Testing Accuracy 0.979 ,Training Accuracy 0.9898546
Iter 11 ,Testing Accuracy 0.9777 ,Training Accuracy 0.9906545
Iter 12 ,Testing Accuracy 0.9777 ,Training Accuracy 0.9918
Iter 13 ,Testing Accuracy 0.978 ,Training Accuracy 0.9911818
Iter 14 ,Testing Accuracy 0.9815 ,Training Accuracy 0.99349093
Iter 15 ,Testing Accuracy 0.9809 ,Training Accuracy 0.99407274
Iter 16 ,Testing Accuracy 0.9808 ,Training Accuracy 0.9940364
Iter 17 ,Testing Accuracy 0.982 ,Training Accuracy 0.9948182
Iter 18 ,Testing Accuracy 0.9807 ,Training Accuracy 0.9947091
Iter 19 ,Testing Accuracy 0.9801 ,Training Accuracy 0.9948182
Iter 20 ,Testing Accuracy 0.9808 ,Training Accuracy 0.99523634
Iter 21 ,Testing Accuracy 0.9818 ,Training Accuracy 0.9956909
Iter 22 ,Testing Accuracy 0.9819 ,Training Accuracy 0.9959091
Iter 23 ,Testing Accuracy 0.9816 ,Training Accuracy 0.9962182
Iter 24 ,Testing Accuracy 0.9815 ,Training Accuracy 0.99625456
Iter 25 ,Testing Accuracy 0.9824 ,Training Accuracy 0.99645454
Iter 26 ,Testing Accuracy 0.9816 ,Training Accuracy 0.9964727
Iter 27 ,Testing Accuracy 0.9826 ,Training Accuracy 0.9962182
Iter 28 ,Testing Accuracy 0.9829 ,Training Accuracy 0.9967818
Iter 29 ,Testing Accuracy 0.9836 ,Training Accuracy 0.9968909
Iter 30 ,Testing Accuracy 0.9834 ,Training Accuracy 0.9969818

