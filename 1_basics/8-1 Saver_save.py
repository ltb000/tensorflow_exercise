
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# In[2]:


# 载入数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 每个批次的大小
batch_size = 50
# 计算一共有多少个批次
n_batch = mnist.train.num_examples//batch_size

# 定义占位符
x = tf.placeholder(tf.float32, [None,784])
y = tf.placeholder(tf.float32, [None,10])

# 创建一个简单的神经网络
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,W)+b)

# 二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))
# 交叉熵
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
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

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(31):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})
            
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print('Iter',epoch,',Testing Accuracy',acc)
        
    #保存模型
    saver.save(sess, 'net/my_net.ckpt')


# In[ ]:


Iter 0 ,Testing Accuracy 0.8919
Iter 1 ,Testing Accuracy 0.9042
Iter 2 ,Testing Accuracy 0.9099
Iter 3 ,Testing Accuracy 0.9137
Iter 4 ,Testing Accuracy 0.9167
Iter 5 ,Testing Accuracy 0.9182
Iter 6 ,Testing Accuracy 0.9186
Iter 7 ,Testing Accuracy 0.9198
Iter 8 ,Testing Accuracy 0.92
Iter 9 ,Testing Accuracy 0.9207
Iter 10 ,Testing Accuracy 0.9209
Iter 11 ,Testing Accuracy 0.923
Iter 12 ,Testing Accuracy 0.9228
Iter 13 ,Testing Accuracy 0.9239
Iter 14 ,Testing Accuracy 0.9245
Iter 15 ,Testing Accuracy 0.9239
Iter 16 ,Testing Accuracy 0.9248
Iter 17 ,Testing Accuracy 0.9249
Iter 18 ,Testing Accuracy 0.9256
Iter 19 ,Testing Accuracy 0.9257
Iter 20 ,Testing Accuracy 0.9263
Iter 21 ,Testing Accuracy 0.9272
Iter 22 ,Testing Accuracy 0.9281
Iter 23 ,Testing Accuracy 0.9283
Iter 24 ,Testing Accuracy 0.9271
Iter 25 ,Testing Accuracy 0.928
Iter 26 ,Testing Accuracy 0.9267
Iter 27 ,Testing Accuracy 0.9285
Iter 28 ,Testing Accuracy 0.9278
Iter 29 ,Testing Accuracy 0.9267
Iter 30 ,Testing Accuracy 0.9284

