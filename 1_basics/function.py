
# coding: utf-8

# In[9]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# In[8]:


x = np.linspace(-5, 5, 200)
y1 = tf.nn.tanh(x)
y2 = tf.nn.sigmoid(x)
y3 = tf.nn.relu(x)

sess = tf.InteractiveSession()

y1,y2,y3 = sess.run([y1,y2,y3])
plt.plot(x,y1,'r')
plt.plot(x,y2,'b')
plt.plot(x,y3,'k')


# In[29]:


x = np.array([5,4,1])
xx = np.array([[1,2,3],[100,5,6],[7,8,9]])
y = np.array([3,7,5])
x_argmax = tf.argmax(x)
y_argmax = tf.argmax(y)
# print(x_argmax)
print(xx)
with tf.Session() as sess:
#     print(sess.run(tf.equal(x_argmax,y_argmax)))
    print(sess.run(tf.argmax(xx,1)))


# In[31]:


x = np.array([True,False,False,False])
print(x)
with tf.Session() as sess:
    print(sess.run(tf.cast(x,tf.float32)))


# In[11]:


x = np.array([1,2,3])
y = np.array([2,3,4])
xx = tf.Variable(np.array([1,2,3]))
yy = tf.Variable(np.array([2,3,4]))
# xy = xx*yy
# print(a.dtpye)
# print(xx*yy)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
#     a = sess.run(xx*yy)
#     print(a)
    print(sess.run(xx).dtype)


# In[12]:


# 声明一个先进先出的队列，队列中最多100个元素，类型为实数。
queue = tf.FIFOQueue(100,'float')
# 定义队列的入队操作。
enqueue_op = queue.enqueue([1,2,3])


# In[36]:


x = []
for i in range(100000):
    x.append([i,i])

x = np.array(x)
print(x.shape)


# In[41]:


a = tf.constant(1)
b = tf.constant(2)
c = tf.Variable(1)
a = a + b
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(a.eval())
    print(a.eval())
    print(c.eval())
    print(c.eval())


# In[48]:


sess = tf.InteractiveSession()

# x = tf.one_hot(2,depth=5)
x = tf.range(12)
reshape = tf.reshape(x,[3,2,2])
transpose = tf.transpose(reshape,[1,2,0])
print(x.eval())
print(reshape.eval())
print(transpose.eval())

# tf.strided_slice()


# In[120]:


a = tf.constant([[0.,0.5],[0.5,1]],dtype=tf.float32)
a = tf.reshape(a,[-1])
print(a.eval())
aa = tf.image.convert_image_dtype(a,tf.uint8)
print(aa.eval())


# In[114]:


a1 = np.arange(3*2*4)
aa1 = tf.reshape(a1,[-1])
a1_argmax = tf.argmax(aa1)
print(aa1.eval().shape)
print(a1_argmax.eval())


# In[119]:


a = tf.one_hot(3,depth=5,dtype=tf.float32)
print(a.eval().dtype)


# In[78]:


from array import array


# In[111]:


a = np.arange(5.).reshape(1,-1)
arr = array('f')
print(a)
arr.extend(a)
arr.extend(a)
arr.extend(a)
aa = np.frombuffer(arr,dtype=np.float32).reshape(-1,5)
print(aa)


# In[133]:


a = tf.constant([1,2],dtype=tf.uint8)
b = tf.constant([2,4],dtype=tf.float64)
c = tf.nn.softmax_cross_entropy_with_logits(labels=a, logits=b)
print(a.eval().dtype,b.eval().dtype,c.eval().dtype)


# In[139]:


a = tf.zeros([3])
print(a.eval().shape)


# In[10]:


def func(a=3):
    print(a)
    
func(a=5)


# In[5]:


for i in range(1,1):
    print(i)

