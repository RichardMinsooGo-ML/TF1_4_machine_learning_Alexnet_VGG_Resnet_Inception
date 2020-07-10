import tensorflow as tf
import numpy as np

# CIFAR-10 데이터를 다운로드 받기 위한 keras의 helper 함수인 load_data 함수를 임포트합니다.
from tensorflow.keras.datasets.cifar100 import load_data
# 최신 Windows Laptop에서만 사용할것.CPU Version이 높을때 사용.
# AVX를 지원하는 CPU는 Giuthub: How to compile tensorflow using SSE4.1, SSE4.2, and AVX. 
# Ubuntu와 MacOS는 지원하지만 Windows는 없었음. 2018-09-29
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Compuntational Graph Initialization
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Network Parameters
INPUT_SIZE  = 784     # MNIST data input (img shape: 28*28)
OUTPUT_SIZE = 100     # MNIST total classes (0-9 digits)
keep_prob   = 0.7     # Dropout, probability to keep units

N_EPISODES = 20
batch_size = 100

# 다음 배치를 읽어오기 위한 next_batch 유틸리티 함수를 정의합니다.
def next_batch(num, data, labels):
    '''
    `num` 개수 만큼의 랜덤한 샘플들과 레이블들을 리턴합니다.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

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

########## set net parameters           ##########
def weight_var(name, shape):
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def bias_var(name, shape):
    return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(0))

weights={
    'WEI_CONV_01': weight_var('WEI_CONV_01',[11,11,3,96]),
    'WEI_CONV_02': weight_var('WEI_CONV_02',[5,5,96,256]),
    'WEI_CONV_03': weight_var('WEI_CONV_03',[3,3,256,384]),
    'WEI_CONV_04': weight_var('WEI_CONV_04',[3,3,384,384]),
    'WEI_CONV_05': weight_var('WEI_CONV_05',[3,3,384,256]),
    'WEI_DENS_01': weight_var('WEI_DENS_01',[4*4*256,4096]),
    'WEI_DENS_02': weight_var('WEI_DENS_02',[4096,4096]),
    'out_w': weight_var('out_w',[4096,OUTPUT_SIZE])
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

def ALEXNET_2(X_Input, weights, biases, keep_prob):
    #### reshape input picture ####
    x=tf.reshape(X_Input, shape=[-1,32,32,3])

    with tf.name_scope('Conv_Layer_01'):
        # CONVOLUTION Layer
        CONV_01     = CONVOLUTION('CONV_01', x, weights['WEI_CONV_01'], biases['BIA_CONV_01'], padding='SAME')
        # Max Pooling (down-sampling)
        MAX_POOL_01 = POOLING('MAX_POOL_01',CONV_01,k=3, s=2, padding='SAME')
        # Apply Normalization
        NORM_01     = NORMALIZATION('NORM_01', MAX_POOL_01, lsize=5)

    with tf.name_scope('Conv_Layer_02'):
        # CONVOLUTION Layer
        CONV_02     = CONVOLUTION('CONV_02', NORM_01, weights['WEI_CONV_02'], biases['BIA_CONV_02'], padding='SAME')
        # Max Pooling (down-sampling)
        MAX_POOL_02 = POOLING('MAX_POOL_02',CONV_02,k=3, s=2, padding='SAME')
        # Apply Normalization
        NORM_02     = NORMALIZATION('NORM_02', MAX_POOL_02, lsize=5)

    with tf.name_scope('Conv_Layer_03'):
        # CONVOLUTION Layer
        CONV_03     = CONVOLUTION('CONV_03', NORM_02, weights['WEI_CONV_03'], biases['BIA_CONV_03'], padding='SAME')

    with tf.name_scope('Conv_Layer_04'):
        # CONVOLUTION Layer
        CONV_04     = CONVOLUTION('CONV_04', CONV_03, weights['WEI_CONV_04'], biases['BIA_CONV_04'], padding='SAME')

    with tf.name_scope('Conv_Layer_05'):
        # CONVOLUTION Layer
        CONV_05     = CONVOLUTION('CONV_05', CONV_04, weights['WEI_CONV_05'], biases['BIA_CONV_05'], padding='SAME')
        # Max Pooling (down-sampling)
        MAX_POOL_05 = POOLING('MAX_POOL_05',CONV_05,k=3, s=2, padding='SAME')
        # Apply Normalization
        NORM_05     = NORMALIZATION('NORM_05', MAX_POOL_05, lsize=5)

    with tf.name_scope('Dense_Layer_01'):
        # Fully connected layer
        DENSE_01=tf.reshape(NORM_05,[-1,weights['WEI_DENS_01'].get_shape().as_list()[0]])
        DENSE_01=tf.add(tf.matmul(DENSE_01,weights['WEI_DENS_01']),biases['BIA_DENS_01'])
        DENSE_01=tf.nn.relu(DENSE_01)

        ## dropout ##
        DENSE_01=tf.nn.dropout(DENSE_01, keep_prob)

    with tf.name_scope('Dense_Layer_02'):
        #### 2 fc ####
        #DENSE_02=tf.reshape(DENSE_01,[-1,weights['WEI_DENS_02'].get_shape().as_list()[0]])
        DENSE_02=tf.add(tf.matmul(DENSE_01,weights['WEI_DENS_02']),biases['BIA_DENS_02'])
        DENSE_02=tf.nn.relu(DENSE_02)

        ## dropout ##
        DENSE_02=tf.nn.dropout(DENSE_02, keep_prob)

    with tf.name_scope('Output_Layer'):
        #### output ####
        logits = tf.add(tf.matmul(DENSE_02,weights['out_w']),biases['out_b'])
        y_pred = tf.nn.softmax(logits)
    return y_pred, logits
"""
# CNN 모델을 정의합니다. 
def ALEXNET_2(x):
    # 입력 이미지
    x_image = x

    # 첫번째 convolutional layer - 하나의 grayscale 이미지를 64개의 특징들(feature)으로 맵핑(maping)합니다.
    W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 64], stddev=5e-2))
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[64]))
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

    # 첫번째 Pooling layer
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 두번째 convolutional layer - 32개의 특징들(feature)을 64개의 특징들(feature)로 맵핑(maping)합니다.
    W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 64, 64], stddev=5e-2))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

    # 두번째 pooling layer.
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 세번째 convolutional layer
    W_conv3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], stddev=5e-2))
    b_conv3 = tf.Variable(tf.constant(0.1, shape=[128]))
    h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)

    # 네번째 convolutional layer
    W_conv4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2))
    b_conv4 = tf.Variable(tf.constant(0.1, shape=[128])) 
    h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, W_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv4)

    # 다섯번째 convolutional layer
    W_conv5 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2))
    b_conv5 = tf.Variable(tf.constant(0.1, shape=[128]))
    h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, W_conv5, strides=[1, 1, 1, 1], padding='SAME') + b_conv5)

    # Fully Connected Layer 1 - 2번의 downsampling 이후에, 우리의 32x32 이미지는 8x8x128 특징맵(feature map)이 됩니다.
    # 이를 384개의 특징들로 맵핑(maping)합니다.
    W_fc1 = tf.Variable(tf.truncated_normal(shape=[8 * 8 * 128, 384], stddev=5e-2))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[384]))

    h_conv5_flat = tf.reshape(h_conv5, [-1, 8*8*128])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)

    # Dropout - 모델의 복잡도를 컨트롤합니다. 특징들의 co-adaptation을 방지합니다.
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) 

    # Fully Connected Layer 2 - 384개의 특징들(feature)을 10개의 클래스-airplane, automobile, bird...-로 맵핑(maping)합니다.
    W_fc2 = tf.Variable(tf.truncated_normal(shape=[384, 10], stddev=5e-2))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
    logits = tf.matmul(h_fc1_drop,W_fc2) + b_fc2
    y_pred = tf.nn.softmax(logits)

    return y_pred, logits
"""

# 인풋 아웃풋 데이터, 드롭아웃 확률을 입력받기위한 플레이스홀더를 정의합니다.
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE])
keep_prob = tf.placeholder(tf.float32)

# CIFAR-10 데이터를 다운로드하고 데이터를 불러옵니다.
(X_train, Y_train), (X_test, Y_test) = load_data()

# scalar 형태의 레이블(0~9)을 One-hot Encoding 형태로 변환합니다.
Y_train_one_hot = tf.squeeze(tf.one_hot(Y_train, OUTPUT_SIZE),axis=1)
Y_test_one_hot = tf.squeeze(tf.one_hot(Y_test, OUTPUT_SIZE),axis=1)

# Convolutional Neural Networks(CNN) 그래프를 생성합니다.
y_pred, logits = ALEXNET_2(x, weights, biases, keep_prob)

# Cross Entropy를 비용함수(loss function)으로 정의하고, RMSPropOptimizer를 이용해서 비용 함수를 최소화합니다.
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

# 정확도를 계산하는 연산을 추가합니다.
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 세션을 열어 실제 학습을 진행합니다.
with tf.Session() as sess:
    
    # 모든 변수들을 초기화한다. 
    sess.run(tf.global_variables_initializer())
    # Total_batch = int(X_train.shape[0]/batch_size)
    Total_batch = 64
    # 10000 Step만큼 최적화를 수행합니다.
    for episode in range(N_EPISODES):
        total_cost = 0
        for i in range(Total_batch):
            batch = next_batch(batch_size, X_train, Y_train_one_hot.eval())

            # 100 Step마다 training 데이터셋에 대한 정확도와 loss를 출력합니다.
            
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
            loss_print = loss.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})

                
            # 20% 확률의 Dropout을 이용해서 학습을 진행합니다.
            sess.run(optimizer, feed_dict={x: batch[0], y: batch[1], keep_prob: 0.8})
            total_cost += loss_print
        print("Epoch: %6d, Loss: %2.6f" % (episode+1, total_cost/Total_batch))

    # 학습이 끝나면 테스트 데이터(10000개)에 대한 정확도를 출력합니다.  
    test_accuracy = 0.0  
    for i in range(10):
        test_batch = next_batch(1000, X_test, Y_test_one_hot.eval())
        test_accuracy = test_accuracy + accuracy.eval(feed_dict={x: test_batch[0], y: test_batch[1], keep_prob: 1.0})
    test_accuracy = test_accuracy / 10;
    print("Test Data Accuracy: %2.4f" % test_accuracy)

