
# coding: utf-8

# In[4]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# In[11]:


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

# 创建一个简单的神经网络
# W = tf.Variable(tf.zeros([784,10]))
# b = tf.Variable(tf.zeros([10]))
#
W1 = tf.Variable(tf.truncated_normal([784,100], stddev=0.1))
b1 = tf.Variable(tf.zeros([100])+0.1)
# L1 = tf.nn.tanh(tf.matmul(x,W1)+b1)
L1 = tf.nn.relu(tf.matmul(x,W1)+b1)
L1_drop = tf.nn.dropout(L1,keep_prob)

W2 = tf.Variable(tf.truncated_normal([100,10], stddev=0.1))
b2 = tf.Variable(tf.zeros([10])+0.1)
#
prediction = tf.nn.softmax(tf.matmul(L1_drop,W2)+b2)

# 二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
# 交叉熵
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# loss = -tf.reduce_sum(y*tf.log(prediction))
# 使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

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
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys, keep_prob:0.7})
            
        test_acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
        train_acc = sess.run(accuracy, feed_dict={x:mnist.train.images, y:mnist.train.labels, keep_prob:1.0})
        print('Iter',epoch,',Testing Accuracy',test_acc,',Training Accuracy',train_acc)


# In[ ]:


Iter 0 ,Testing Accuracy 0.8419 ,Training Accuracy 0.84049094
Iter 1 ,Testing Accuracy 0.9389 ,Training Accuracy 0.93912727
Iter 2 ,Testing Accuracy 0.9484 ,Training Accuracy 0.9536
Iter 3 ,Testing Accuracy 0.9554 ,Training Accuracy 0.96076363
Iter 4 ,Testing Accuracy 0.9594 ,Training Accuracy 0.9664182
Iter 5 ,Testing Accuracy 0.9632 ,Training Accuracy 0.97043633
Iter 6 ,Testing Accuracy 0.9653 ,Training Accuracy 0.97278184
Iter 7 ,Testing Accuracy 0.9695 ,Training Accuracy 0.97794545
Iter 8 ,Testing Accuracy 0.967 ,Training Accuracy 0.97696364
Iter 9 ,Testing Accuracy 0.9702 ,Training Accuracy 0.9796182
Iter 10 ,Testing Accuracy 0.9704 ,Training Accuracy 0.9809091
Iter 11 ,Testing Accuracy 0.9668 ,Training Accuracy 0.98
Iter 12 ,Testing Accuracy 0.9707 ,Training Accuracy 0.9827818
Iter 13 ,Testing Accuracy 0.9727 ,Training Accuracy 0.9854182
Iter 14 ,Testing Accuracy 0.9749 ,Training Accuracy 0.98634547
Iter 15 ,Testing Accuracy 0.9735 ,Training Accuracy 0.9872909
Iter 16 ,Testing Accuracy 0.9744 ,Training Accuracy 0.9876909
Iter 17 ,Testing Accuracy 0.9742 ,Training Accuracy 0.98841816
Iter 18 ,Testing Accuracy 0.9756 ,Training Accuracy 0.9887818
Iter 19 ,Testing Accuracy 0.975 ,Training Accuracy 0.98905456
Iter 20 ,Testing Accuracy 0.9748 ,Training Accuracy 0.98954546
Iter 21 ,Testing Accuracy 0.975 ,Training Accuracy 0.9896182
Iter 22 ,Testing Accuracy 0.9756 ,Training Accuracy 0.99012727
Iter 23 ,Testing Accuracy 0.9757 ,Training Accuracy 0.99036366
Iter 24 ,Testing Accuracy 0.9766 ,Training Accuracy 0.99054545
Iter 25 ,Testing Accuracy 0.9763 ,Training Accuracy 0.9906727
Iter 26 ,Testing Accuracy 0.9756 ,Training Accuracy 0.9908
Iter 27 ,Testing Accuracy 0.9762 ,Training Accuracy 0.99112725
Iter 28 ,Testing Accuracy 0.9765 ,Training Accuracy 0.9912909
Iter 29 ,Testing Accuracy 0.9756 ,Training Accuracy 0.9913091
Iter 30 ,Testing Accuracy 0.9765 ,Training Accuracy 0.9914182
Iter 31 ,Testing Accuracy 0.9764 ,Training Accuracy 0.9914727
Iter 32 ,Testing Accuracy 0.976 ,Training Accuracy 0.9916
Iter 33 ,Testing Accuracy 0.9761 ,Training Accuracy 0.99167275
Iter 34 ,Testing Accuracy 0.9764 ,Training Accuracy 0.99167275
Iter 35 ,Testing Accuracy 0.9771 ,Training Accuracy 0.99183637
Iter 36 ,Testing Accuracy 0.9768 ,Training Accuracy 0.9919818
Iter 37 ,Testing Accuracy 0.9771 ,Training Accuracy 0.99203634
Iter 38 ,Testing Accuracy 0.9777 ,Training Accuracy 0.9920909
Iter 39 ,Testing Accuracy 0.9764 ,Training Accuracy 0.9921091
Iter 40 ,Testing Accuracy 0.9775 ,Training Accuracy 0.9922
Iter 41 ,Testing Accuracy 0.977 ,Training Accuracy 0.9923091
Iter 42 ,Testing Accuracy 0.9769 ,Training Accuracy 0.9923273
Iter 43 ,Testing Accuracy 0.977 ,Training Accuracy 0.99236363
Iter 44 ,Testing Accuracy 0.9775 ,Training Accuracy 0.9924
Iter 45 ,Testing Accuracy 0.9767 ,Training Accuracy 0.9924545
Iter 46 ,Testing Accuracy 0.977 ,Training Accuracy 0.9924909
Iter 47 ,Testing Accuracy 0.9766 ,Training Accuracy 0.9925454
Iter 48 ,Testing Accuracy 0.9772 ,Training Accuracy 0.9926
Iter 49 ,Testing Accuracy 0.9772 ,Training Accuracy 0.9926182
Iter 50 ,Testing Accuracy 0.9775 ,Training Accuracy 0.9926364

