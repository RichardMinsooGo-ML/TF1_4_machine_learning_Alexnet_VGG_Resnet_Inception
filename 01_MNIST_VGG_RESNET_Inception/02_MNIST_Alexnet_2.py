import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
# 최신 Windows Laptop에서만 사용할것.CPU Version이 높을때 사용.
# AVX를 지원하는 CPU는 Giuthub: How to compile tensorflow using SSE4.1, SSE4.2, and AVX. 
# Ubuntu와 MacOS는 지원하지만 Windows는 없었음. 2018-09-29
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Compuntational Graph Initialization
from tensorflow.python.framework import ops
ops.reset_default_graph()

DATA_DIR = "/tmp/ML/MNIST_data"
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

# Hyper Parameters
Alpha_Lr       = 0.001   # Learning Rate Alpha
training_iters = 2000
batch_size     = 100
display_step   = 25
N_EPISODES     = 10

# Network Parameters
INPUT_SIZE  = 784    # MNIST data input (img shape: 28*28)
OUTPUT_SIZE = 10     # MNIST total classes (0-9 digits)
keep_prob   = 0.7    # Dropout, probability to keep units

# tf Graph input
X_Input = tf.placeholder(tf.float32, [None, INPUT_SIZE])
Y_Output = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

# Create AlexNet model
########## define conv process          ##########
def CONVOLUTION(name,X_Input,W,b,strides=1, padding='SAME'):
    X_Input = tf.nn.conv2d(X_Input,W,strides=[1,strides,strides,1],padding=padding)
    X_Input = tf.nn.bias_add(X_Input,b)
    return tf.nn.relu(X_Input,name=name)

########## define pool process          ##########
def POOLING(name, X_Input, k=3, s=2, padding='SAME'):
    return tf.nn.max_pool(X_Input,ksize=[1,k,k,1],strides=[1,s,s,1],padding=padding,name=name)

########## define NORMALIZATION process ##########
def NORMALIZATION(name, l_input, lsize=5):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.0001, beta=0.75, name=name)

########## set net parameters ##########
def weight_var(name, shape):
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def bias_var(name, shape):
    return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(0))

weights={
    'WEI_CONV_01': weight_var('WEI_CONV_01',[11,11,1,96]),
    'WEI_CONV_02': weight_var('WEI_CONV_02',[5,5,96,256]),
    'WEI_CONV_03': weight_var('WEI_CONV_03',[3,3,256,384]),
    'WEI_CONV_04': weight_var('WEI_CONV_04',[3,3,384,384]),
    'WEI_CONV_05': weight_var('WEI_CONV_05',[3,3,384,256]),
    'WEI_DENS_01': weight_var('WEI_DENS_01',[4*4*256,4096]),
    'WEI_DENS_02': weight_var('WEI_DENS_02',[4096,4096]),
    'out_w': weight_var('out_w',[4096,10])
}
biases={
    'BIA_CONV_01': bias_var('BIA_CONV_01',[96]),
    'BIA_CONV_02': bias_var('BIA_CONV_02',[256]),
    'BIA_CONV_03': bias_var('BIA_CONV_03',[384]),
    'BIA_CONV_04': bias_var('BIA_CONV_04',[384]),
    'BIA_CONV_05': bias_var('BIA_CONV_05',[256]),
    'BIA_DENS_01': bias_var('BIA_DENS_01',[4096]),
    'BIA_DENS_02': bias_var('BIA_DENS_02',[4096]),
    'out_b': bias_var('out_b',[OUTPUT_SIZE])
}

"""
########## set net parameters ##########
weights = {
    'WEI_CONV_01': tf.Variable(tf.random_normal([11,11,1,96])),
    'WEI_CONV_02': tf.Variable(tf.random_normal([5,5,96,256])),
    'WEI_CONV_03': tf.Variable(tf.random_normal([3,3,256,384])),
    'WEI_CONV_04': tf.Variable(tf.random_normal([3,3,384,384])),
    'WEI_CONV_05': tf.Variable(tf.random_normal([3,3,384,256])),
    'WEI_DENS_01': tf.Variable(tf.random_normal([4*4*256,4096])),
    'WEI_DENS_02': tf.Variable(tf.random_normal([4096,4096])),
    'out_w': tf.Variable(tf.random_normal([4096,10]))
}
biases = {
    'BIA_CONV_01': tf.Variable(tf.random_normal([96])),
    'BIA_CONV_02': tf.Variable(tf.random_normal([256])),
    'BIA_CONV_03': tf.Variable(tf.random_normal([384])),
    'BIA_CONV_04': tf.Variable(tf.random_normal([384])),
    'BIA_CONV_05': tf.Variable(tf.random_normal([256])),
    'BIA_DENS_01': tf.Variable(tf.random_normal([4096])),
    'BIA_DENS_02': tf.Variable(tf.random_normal([4096])),
    'out_b': tf.Variable(tf.random_normal([OUTPUT_SIZE]))
}
"""

##################### build net model ##########################

########## define net structure ##########
def ALEX_NET(X_Input, weights, biases, keep_prob):
    #### reshape input picture ####
    x=tf.reshape(X_Input, shape=[-1,28,28,1])

    # CONVOLUTION Layer
    CONV_01     = CONVOLUTION('CONV_01', x, weights['WEI_CONV_01'], biases['BIA_CONV_01'], padding='SAME')
    # Max Pooling (down-sampling)
    MAX_POOL_01 = POOLING('MAX_POOL_01',CONV_01,k=3, s=2, padding='SAME')
    # Apply Normalization
    NORM_01     = NORMALIZATION('NORM_01', MAX_POOL_01, lsize=5)

    # CONVOLUTION Layer
    CONV_02     = CONVOLUTION('CONV_02', NORM_01, weights['WEI_CONV_02'], biases['BIA_CONV_02'], padding='SAME')
    # Max Pooling (down-sampling)
    MAX_POOL_02 = POOLING('MAX_POOL_02',CONV_02,k=3, s=2, padding='SAME')
    # Apply Normalization
    NORM_02     = NORMALIZATION('NORM_02', MAX_POOL_02, lsize=5)

    # CONVOLUTION Layer
    CONV_03     = CONVOLUTION('CONV_03', NORM_02, weights['WEI_CONV_03'], biases['BIA_CONV_03'], padding='SAME')

    # CONVOLUTION Layer
    CONV_04     = CONVOLUTION('CONV_04', CONV_03, weights['WEI_CONV_04'], biases['BIA_CONV_04'], padding='SAME')

    # CONVOLUTION Layer
    CONV_05     = CONVOLUTION('CONV_05', CONV_04, weights['WEI_CONV_05'], biases['BIA_CONV_05'], padding='SAME')
    # Max Pooling (down-sampling)
    MAX_POOL_05 = POOLING('MAX_POOL_05',CONV_05,k=3, s=2, padding='SAME')
    # Apply Normalization
    NORM_05     = NORMALIZATION('NORM_05', MAX_POOL_05, lsize=5)

    # Fully connected layer
    DENSE_01=tf.reshape(NORM_05,[-1,weights['WEI_DENS_01'].get_shape().as_list()[0]])
    DENSE_01=tf.add(tf.matmul(DENSE_01,weights['WEI_DENS_01']),biases['BIA_DENS_01'])
    DENSE_01=tf.nn.relu(DENSE_01)

    ## dropout ##
    DENSE_01=tf.nn.dropout(DENSE_01, keep_prob)

    #### 2 fc ####
    #DENSE_02=tf.reshape(DENSE_01,[-1,weights['WEI_DENS_02'].get_shape().as_list()[0]])
    DENSE_02=tf.add(tf.matmul(DENSE_01,weights['WEI_DENS_02']),biases['BIA_DENS_02'])
    DENSE_02=tf.nn.relu(DENSE_02)

    ## dropout ##
    DENSE_02=tf.nn.dropout(DENSE_02, keep_prob)

    #### output ####
    Pred_m = tf.add(tf.matmul(DENSE_02,weights['out_w']),biases['out_b'])
    return Pred_m

########## define model, loss and optimizer ##########

# Construct model
pred = ALEX_NET(X_Input, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y_Output))
optimizer = tf.train.AdamOptimizer(learning_rate = Alpha_Lr).minimize(cost)

# Evaluate model
is_correct = tf.equal(tf.argmax(pred,1), tf.argmax(Y_Output,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

##################### train and evaluate model ##########################

"""
########## initialize variables ##########
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

step = 1
# total_batch = int(mnist.train.num_examples / batch_size)
total_batch = 64

# train my model
print('Learning started. It takes sometime.')

for episode in range(N_EPISODES+1):

    #### iteration ####
    for i in range(total_batch):

        step += 1

        ##### get X_Input,Y_Output #####
        batch_xs, batch_ys=mnist.train.next_batch(batch_size)

        ##### optimizer ####
        sess.run(optimizer,feed_dict={X_Input:batch_xs, Y_Output:batch_ys})


        ##### show loss and acc ##### 
        if step % display_step==0:
            loss,acc=sess.run([cost, accuracy],feed_dict={X_Input: batch_xs, Y_Output: batch_ys})
            print("episode "+ str(episode) + ", Minibatch Loss=" + \
                "{:.6f}".format(loss) + ", Training Accuracy= "+ \
                "{:.5f}".format(acc))


print("Optimizer Finished!")

##### test accuracy #####
for i in range(total_batch):
    batch_xs,batch_ys=mnist.test.next_batch(batch_size)
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X_Input: batch_xs, Y_Output: batch_ys}))
    
"""
# initialize
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# train my model
print('Learning started. It takes sometime.')


for episode in range(N_EPISODES):
    avg_cost = 0
    # total_batch = int(mnist.train.num_examples / batch_size)
    total_batch = 64

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X_Input: batch_xs, Y_Output: batch_ys, keep_prob: 0.7}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('episode:', '%06d' % (episode + 1), 'cost =', '{:.5f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy

is_correct = tf.equal(tf.argmax(pred, 1), tf.argmax(Y_Output, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('Accuracy:', '{:.5f}'.format(sess.run(accuracy, feed_dict={X_Input: mnist.test.images, Y_Output: mnist.test.labels, keep_prob: 1})))

# Get one and predict
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", '{:.5f}'.format(sess.run(tf.argmax(Pred_m, 1), feed_dict={X_Input: mnist.test.images[r:r + 1], keep_prob: 1})))

#########
# 결과 확인 (matplot)
######
labels = sess.run(pred,
                  feed_dict={X_Input: mnist.test.images,
                             Y_Output: mnist.test.labels,
                             keep_prob: 1})

fig = plt.figure()
for i in range(60):
    subplot = fig.add_subplot(4, 15, i + 1)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_title('%d' % np.argmax(labels[i]))
    subplot.imshow(mnist.test.images[i].reshape((28, 28)),
                   cmap=plt.cm.gray_r)

plt.show()


# 세션을 닫습니다.
sess.close()


# Step 10. Tune hyperparameters:
# Step 11. Deploy/predict new outcomes:

"""
"""

