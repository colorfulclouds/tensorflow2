
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt


# In[2]:


def get_sample():
    x = np.linspace(-1 , 1 , 300)[: , np.newaxis]
    noise = np.random.normal(0 , 0.05 , size = x.shape)
    y = np.multiply(np.square(x),x) + noise
    
    return x , y


# In[3]:


x_data , y_data = get_sample()


# In[4]:


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data , y_data)

plt.ion()
plt.show(block = False)


# In[31]:


xs = tf.placeholder(tf.float32 , shape = [None , 1])
ys = tf.placeholder(tf.float32 , shape = [None , 1])

weights_1 = tf.Variable(tf.random_normal([1,5]))
biases_1 = tf.Variable(tf.zeros([5]))

layer_1 = tf.nn.sigmoid(tf.matmul(xs , weights_1) + biases_1)

weights_2 = tf.Variable(tf.random_normal([5 , 1]))
biases_2 = tf.Variable(tf.zeros([1]))
prediction = tf.matmul(layer_1 , weights_2) + biases_2

loss = tf.reduce_mean(tf.square(ys - prediction))

train_step = tf.train.GradientDescentOptimizer(0.6).minimize(loss)


# In[37]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(5000):
	sess.run(train_step , feed_dict={xs:x_data , ys:y_data})
	p = sess.run(prediction , feed_dict={xs:x_data , ys:y_data})
	if i%1000 == 0:
		try:
			ax.lines.remove(lines[0])
		except Exception:
			pass
			
		lines = ax.plot(x_data , p , 'r-' , lw = 5)
		plt.pause(0.1)
    
#print(sess.run(loss , feed_dict={xs:x_data , ys:y_data}))

# In[38]:


x_test=np.array([[0.9]])
print(sess.run(prediction , feed_dict={xs:x_test}))


# In[25]:


y_data

