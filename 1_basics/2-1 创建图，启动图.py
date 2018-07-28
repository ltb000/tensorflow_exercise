
# coding: utf-8

# In[18]:


import tensorflow as tf

m1 = tf.constant([[3,3]])
m2 = tf.constant([[2],[3]])
m = tf.matmul(m1,m2)
#m = m1 * m2


# In[19]:


sess = tf.Session()
result = sess.run(m)
print(result)
sess.close()


# In[20]:


with tf.Session() as sess:
    result = sess.run(m)
    print(result)

