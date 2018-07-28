
# coding: utf-8

# In[2]:


import tensorflow as tf


# In[7]:


x = tf.Variable([1,2])
a = tf.constant([3,3])

sub = tf.subtract(x,a)

add = tf.add(x,sub)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))


# In[4]:


state = tf.Variable(0,name='counter')
new_value = tf.add(state,1)
update = tf.assign(state,new_value)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))
#         print(sess.run(update))


# In[7]:


a = 0

with tf.Session() as sess:
    for step in range(5):
        a = tf.add(a,1)
        print(sess.run(a))

