
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector


# In[2]:


# 载入数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# # 运行次数
# steps = 1001
# 图片数量
image_num = 3000
# 文件路径
DIR = 'C:/Users/56898/AnacondaProjects/tensorflow/project/'
# # 每个批次的大小
batch_size = 50
# 计算一共有多少个批次
n_batch = mnist.train.num_examples//batch_size

# 定义会话
# sess = tf.Session()

# 参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)#平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev',stddev)#标准差
        tf.summary.scalar('max',tf.reduce_max(var))#最大值
        tf.summary.scalar('min',tf.reduce_min(var))#最小值
        tf.summary.histogram('histogram',var)#直方图
        
# 命名空间
with tf.name_scope('input'):
    # 定义占位符
    x = tf.placeholder(tf.float32, [None,784], name='x_input')
    y = tf.placeholder(tf.float32, [None,10], name='y_input')

# 显示图片
with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x,[-1,28,28,1])
    tf.summary.image('input', image_shaped_input, 10)

with tf.name_scope('layer'):
    # 创建一个简单的神经网络
    with tf.name_scope('wights'):
        W = tf.Variable(tf.zeros([784,10]))
        variable_summaries(W)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]))
        variable_summaries(b)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x,W)+b
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b)

with tf.name_scope('loss'):
    # 二次代价函数
    loss = tf.reduce_mean(tf.square(y-prediction))
    tf.summary.scalar('loss',loss)
    # 交叉熵
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    # loss = -tf.reduce_sum(y*tf.log(prediction))
with tf.name_scope('train'):
    # 使用梯度下降法
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)


with tf.name_scope('accuracy'):
    # 预测准确率 
    with tf.name_scope('correct_prediction'):
        # 先生成一个布尔型列表中
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(prediction,1))
    with tf.name_scope('accuracy'):
        # 求准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy',accuracy)
        
# 合并所有的summary
merged = tf.summary.merge_all()

# 产生metadata文件
np.savetxt(DIR+'metadata.tsv', np.argmax(mnist.test.labels[:image_num], 1).reshape(image_num,1))
# 载入图片
embedding_var = tf.Variable(mnist.test.images[:image_num], trainable=False, name='embedding')

# 初始化变量
init = tf.global_variables_initializer()

run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter(DIR, sess.graph)
    
    saver = tf.train.Saver()
    
    config = projector.ProjectorConfig()

    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name

    embedding.metadata_path = DIR+'metadata.tsv'

    embedding.sprite.image_path = DIR+'mnist_10k_sprite.png'
    embedding.sprite.single_image_dim.extend([28,28])

    projector.visualize_embeddings(writer,config)
    
    
    for i in range(11*n_batch):
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        if i%n_batch == 0:
            acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
            print('Iter',i//n_batch,',Testing Accuracy',acc,',i=',i,',n_batch=',n_batch)
        
            summary,_ = sess.run([merged,train_step], 
                             feed_dict={x:batch_xs, y:batch_ys},
                             options=run_options,
                             run_metadata=run_metadata)
        
            writer.add_summary(summary,i)
            writer.add_run_metadata(run_metadata, 'step{}'.format(i))
        else:
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})
        
    
    saver.save(sess, DIR+'model.ckpt',global_step=i)
        
    writer.close()

