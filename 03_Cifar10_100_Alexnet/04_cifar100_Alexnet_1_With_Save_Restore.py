import tensorflow as tf
import numpy as np
import time, datetime
import os
# CIFAR-10 데이터를 다운로드 받기 위한 keras의 helper 함수인 load_data 함수를 임포트합니다.
from tensorflow.keras.datasets.cifar100 import load_data

OUTPUT_SIZE = 100
N_EPISODES = 3
batch_size = 100

DIR_Checkpoint  = "/tmp/ML/18_1_Cifar100_ALEXNET_1/CheckPoint"
DIR_Tensorboard = "/tmp/ML/18_1_Cifar100_ALEXNET_1/Tensorboard"

# 학습에 직접적으로 사용하지 않고 학습 횟수에 따라 단순히 증가시킬 변수를 만듭니다.
global_step = tf.Variable(0, trainable=False, name='global_step')

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
def CONVOLUTION(name, X_Input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(X_Input, w, strides=[1, 1, 1, 1], padding='SAME'),b), name=name)

########## define pool process          ##########
def POOLING(name, X_Input, k):
    return tf.nn.max_pool(X_Input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

########## define NORMALIZATION process ##########
def NORMALIZATION(name, X_Input, lsize=4):
    return tf.nn.lrn(X_Input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

########## set net parameters ##########
weights = {
    'WEI_CONV_01': tf.Variable(tf.random_normal([3, 3, 3, 64])),
    'WEI_CONV_02': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    'WEI_CONV_03': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    'WEI_DENS_01': tf.Variable(tf.random_normal([4*4*256, 1024])),
    'WEI_DENS_02': tf.Variable(tf.random_normal([1024, 1024])),
    'out': tf.Variable(tf.random_normal([1024, OUTPUT_SIZE]))
}
biases = {
    'BIA_CONV_01': tf.Variable(tf.random_normal([64])),
    'BIA_CONV_02': tf.Variable(tf.random_normal([128])),
    'BIA_CONV_03': tf.Variable(tf.random_normal([256])),
    'BIA_DENS_01': tf.Variable(tf.random_normal([1024])),
    'BIA_DENS_02': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([OUTPUT_SIZE]))
}

# CNN 모델을 정의합니다. 
def ALEX_NET_1(x):
    # 입력 이미지
    x_image = x

    with tf.name_scope('Hidden_Layer_01'):
        # Convolution Layer
        CONV_01     = CONVOLUTION('CONV_01', x_image, weights['WEI_CONV_01'], biases['BIA_CONV_01'])
        # Max Pooling (down-sampling)
        MAX_POOL_01 = POOLING('MAX_POOL_01', CONV_01, k=2)
        # Apply Normalization
        NORM_01     = NORMALIZATION('NORM_01', MAX_POOL_01, lsize=4)
        # Apply Dropout
        NORM_01     = tf.nn.dropout(NORM_01, keep_prob)

    with tf.name_scope('Hidden_Layer_02'):
        # Convolution Layer
        CONV_02     = CONVOLUTION('CONV_02', NORM_01, weights['WEI_CONV_02'], biases['BIA_CONV_02'])
        # Max Pooling (down-sampling)
        MAX_POOL_02 = POOLING('MAX_POOL_02', CONV_02, k=2)
        # Apply Normalization
        NORM_02     = NORMALIZATION('NORM_02', MAX_POOL_02, lsize=4)
        # Apply Dropout
        NORM_02     = tf.nn.dropout(NORM_02, keep_prob)

    with tf.name_scope('Hidden_Layer_03'):
        # Convolution Layer
        CONV_03     = CONVOLUTION('CONV_03', NORM_02, weights['WEI_CONV_03'], biases['BIA_CONV_03'])
        # Max Pooling (down-sampling)
        MAX_POOL_03 = POOLING('MAX_POOL_03', CONV_03, k=2)
        # Apply Normalization
        NORM_03     = NORMALIZATION('NORM_03', MAX_POOL_03, lsize=4)
        # Apply Dropout
        NORM_03     = tf.nn.dropout(NORM_03, keep_prob)

    with tf.name_scope('Dense_Layer_01'):
        # Fully connected layer
        DENSE_01 = tf.reshape(NORM_03, [-1, weights['WEI_DENS_01'].get_shape().as_list()[0]]) # Reshape CONV_03 output to fit dense layer input
        DENSE_01 = tf.nn.relu(tf.matmul(DENSE_01, weights['WEI_DENS_01']) + biases['BIA_DENS_01'], name='fc1') # Relu activation

    with tf.name_scope('Output_Layer'):
        DENSE_02 = tf.nn.relu(tf.matmul(DENSE_01, weights['WEI_DENS_02']) + biases['BIA_DENS_02'], name='fc2') # Relu activation

        # Output, class prediction
        logits   = tf.matmul(DENSE_02, weights['out']) + biases['out']
        y_pred = tf.nn.softmax(logits)

    return y_pred, logits

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
y_pred, logits = ALEX_NET_1(x)

with tf.name_scope('optimizer'):
    # Cross Entropy를 비용함수(loss function)으로 정의하고, RMSPropOptimizer를 이용해서 비용 함수를 최소화합니다.
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
    optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(loss, global_step=global_step)

# 정확도를 계산하는 연산을 추가합니다.
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 세션을 열어 실제 학습을 진행합니다.
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    # 모든 변수들을 초기화한다. 
    # sess.run(tf.global_variables_initializer())
    start_time = time.time()

    if not os.path.exists(DIR_Checkpoint):
        os.makedirs(DIR_Checkpoint)
    if not os.path.exists(DIR_Tensorboard):
        os.makedirs(DIR_Tensorboard)

    saver = tf.train.Saver(tf.global_variables())

    ckpt = tf.train.get_checkpoint_state(DIR_Checkpoint)

    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Variables are restored!')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    else:
        sess.run(init)
        print('Variables are initialized!')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    # train my model
    print('Learning started. It takes sometime.')
    
    # train my model
    Total_batch = int(X_train.shape[0]/batch_size) 

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
        # print("Epoch: %6d, Loss: %2.6f" % (episode+1, total_cost/Total_batch))
        print('Global Step:', '%07d' % int(sess.run(global_step)/Total_batch), 'cost =', '{:02.6f}'.format(total_cost/Total_batch))
        
        elapsed_time = datetime.timedelta(seconds=int(time.time()-start_time))
        print("[{}]".format(elapsed_time))

    print('Learning Finished!')

    # 최적화가 끝난 뒤, 변수를 저장합니다.
    saver.save(sess, DIR_Checkpoint + './dnn.ckpt', global_step=global_step)

    # 학습이 끝나면 테스트 데이터(10000개)에 대한 정확도를 출력합니다.  
    test_accuracy = 0.0  
    for i in range(10):
        test_batch = next_batch(1000, X_test, Y_test_one_hot.eval())
        test_accuracy = test_accuracy + accuracy.eval(feed_dict={x: test_batch[0], y: test_batch[1], keep_prob: 1.0})
    test_accuracy = test_accuracy / 10;
    print("Test Data Accuracy: %2.4f" % test_accuracy)
    elapsed_time = time.time() - start_time
    formatted = datetime.timedelta(seconds=int(elapsed_time))
    print("=== training time elapsed: {}s ===".format(formatted))
