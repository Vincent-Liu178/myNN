import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


#载入数据集
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#定义每个批次的大小

mini_batch = 200
#计算有多少个批次

n = mnist.train.num_examples//mini_batch

def w_v(shape):
    init = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(init)

def b_v(shape):
    init = tf.constant(0.1,shape = shape)
    return tf.Variable(init)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')





#d定义两个placeholder
x = tf.placeholder(tf.float32,[None,784],name = 'x-input')
y = tf.placeholder(tf.float32,[None,10],name = 'y-input')

#改变x得维度
x_image= tf.reshape(x,[-1,28,28,1])

#创建一个神经网络
W_1 = w_v([5,5,1,32])
b_1 = b_v([32])

h_1 = tf.nn.relu(conv2d(x_image,W_1) + b_1)
h_p1 = max_pool2x2(h_1)

#创建第二个权值和偏重

W_2 = w_v([5,5,32,64])
b_2 = b_v([64])

h_2 = tf.nn.relu(conv2d(h_p1,W_2) + b_2)
h_p2 = max_pool2x2(h_2)

#全连接层
W_f1 = w_v([7*7*64,1024])
b_f1 = b_v([1024])

h_p2_f = tf.reshape(h_p2,[-1,7*7*64])
h_f1 = tf.nn.relu(tf.matmul(h_p2_f,W_f1) + b_f1)

keep_prob = tf.placeholder(tf.float32)
h_f_drop = tf.nn.dropout(h_f1,keep_prob)

#全连ing2
W_f2= w_v([1024,10])
b_f2 = b_v([10])

pre_out = tf.nn.softmax(tf.matmul(h_f_drop,W_f2) + b_f2)
#d定义损失函数

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits = pre_out))

train = tf.train.AdamOptimizer(1e-4).minimize(loss)

#初始化变量

init = tf.global_variables_initializer()

#比较值  返回 True和False
C_pre = tf.equal(tf.argmax(y,1),tf.argmax(pre_out,1))
#求准确率
accuracy = tf.reduce_mean(tf.cast(C_pre,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for i in range(50):
       for t in range(n):
            b_x,b_y = mnist.train.next_batch(mini_batch)  # bx是像素值 by是标签值
            sess.run(train,feed_dict={x:b_x,y:b_y,keep_prob:0.7})

       acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
       print('第'+str(i)+'训练后的准确率为'+str(acc))
