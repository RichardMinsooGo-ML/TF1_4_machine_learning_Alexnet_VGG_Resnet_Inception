import tensorflow as tf
import numpy as np
import time, datetime
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

tf.set_random_seed(777)  # reproducibility

DIR_Checkpoint  = "/tmp/ML/18_4_Cifar10_Inception_Light/CheckPoint"
DIR_Tensorboard = "/tmp/ML/18_4_Cifar10_Inception_Light/Tensorboard"

# 학습에 직접적으로 사용하지 않고 학습 횟수에 따라 단순히 증가시킬 변수를 만듭니다.
global_step = tf.Variable(0, trainable=False, name='global_step')

N_EPISODES = 360
batch_size = 100
keep_prob = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)

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

def conv_bn_activ_dropout(x, n_filters, kernel_size, strides):
    # , strides, dropout_rate, training, seed, padding='SAME', activ_fn=tf.nn.relu):
    #with tf.variable_scope(name):
    # net = tf.layers.conv2d(x, n_filters, kernel_size, strides=1,activation=tf.nn.relu, padding='SAME')
    net = tf.layers.conv2d(x, n_filters, kernel_size, strides,activation=tf.nn.relu, padding='SAME')
    net = tf.layers.batch_normalization(net)
    net = tf.layers.dropout(net, 0.7, is_training)

    return net

def inception_block_a(x, name='inception_a'):
    # num of channels: 96 = 24*4
    with tf.variable_scope(name):
        # with tf.variable_scope("branch1"):
        b1 = tf.layers.average_pooling2d(x, [3,3], 1, padding='SAME')
        b1 = conv_bn_activ_dropout(x = b1, n_filters = 24, kernel_size = [1,1], strides=1)

        # with tf.variable_scope("branch2"):
        b2 = conv_bn_activ_dropout(x = x, n_filters = 24, kernel_size = [1,1], strides=1)

        # with tf.variable_scope("branch3"):
        b3 = conv_bn_activ_dropout(x = x, n_filters = 16, kernel_size = [1,1], strides=1)
        b3 = conv_bn_activ_dropout(x = b3, n_filters = 24, kernel_size = [3,3], strides=1)

        # with tf.variable_scope("branch4"):
        b4 = conv_bn_activ_dropout(x = x, n_filters = 16, kernel_size = [1,1], strides=1)
        b4 = conv_bn_activ_dropout(x = b4, n_filters = 24, kernel_size = [3,3], strides=1)
        b4 = conv_bn_activ_dropout(x = b4, n_filters = 24, kernel_size = [3,3], strides=1)

        concat = tf.concat([b1, b2, b3, b4], axis=-1) # Q. -1 axis works well?
        return concat

def inception_block_b(x, name='inception_b'):
    # num of channels: 256 = 32 + 96 + 64 + 64
    with tf.variable_scope(name):
        b1 = tf.layers.average_pooling2d(x, [3,3], 1, padding='SAME')
        b1 = conv_bn_activ_dropout(x = b1, n_filters = 32, kernel_size = [1,1], strides=1)

        b2 = conv_bn_activ_dropout(x = x, n_filters = 96, kernel_size = [1,1], strides=1)

        b3 = conv_bn_activ_dropout(x = x, n_filters = 48, kernel_size = [1,1], strides=1)
        b3 = conv_bn_activ_dropout(x = b3, n_filters = 56, kernel_size = [1,7], strides=1)
        b3 = conv_bn_activ_dropout(x = b3, n_filters = 64, kernel_size = [7,1], strides=1)

        b4 = conv_bn_activ_dropout(x = x, n_filters = 48, kernel_size = [1,1], strides=1)
        b4 = conv_bn_activ_dropout(x = b4, n_filters = 48, kernel_size = [1,7], strides=1)
        b4 = conv_bn_activ_dropout(x = b4, n_filters = 48, kernel_size = [7,1], strides=1)
        b4 = conv_bn_activ_dropout(x = b4, n_filters = 64, kernel_size = [1,7], strides=1)
        b4 = conv_bn_activ_dropout(x = b4, n_filters = 64, kernel_size = [7,1], strides=1)

        return tf.concat([b1, b2, b3, b4], axis=-1)

def inception_block_c(x, name='inception_c'):
    # num of channels: 384 = 64*6
    with tf.variable_scope(name):
        b1 = tf.layers.average_pooling2d(x, [3,3], 1, padding='SAME')
        b1 = conv_bn_activ_dropout(x = b1, n_filters = 64, kernel_size = [1,1], strides=1)

        b2 = conv_bn_activ_dropout(x = x, n_filters = 64, kernel_size = [1,1], strides=1)

        b3 = conv_bn_activ_dropout(x = x, n_filters = 96, kernel_size = [1,1], strides=1)
        b3_1 = conv_bn_activ_dropout(x = b3, n_filters = 64, kernel_size = [1,3], strides=1)
        b3_2 = conv_bn_activ_dropout(x = b3, n_filters = 64, kernel_size = [3,1], strides=1)

        b4 = conv_bn_activ_dropout(x = x, n_filters = 96, kernel_size = [1,1], strides=1)
        b4 = conv_bn_activ_dropout(x = b4, n_filters = 112, kernel_size = [1,3], strides=1)
        b4 = conv_bn_activ_dropout(x = b4, n_filters = 128, kernel_size = [3,1], strides=1)
        b4_1 = conv_bn_activ_dropout(x = b4, n_filters = 64, kernel_size = [3,1], strides=1)
        b4_2 = conv_bn_activ_dropout(x = b4, n_filters = 64, kernel_size = [1,3], strides=1)

        return tf.concat([b1, b2, b3_1, b3_2, b4_1, b4_2], axis=-1)

# reduction block do downsampling & change ndim of channel
def reduction_block_a(x, name='reduction_a'):
    # 96 => 256 (= 96 + 64 + 96)
    # SAME : 28 > 14 > 7 > 4
    # VALID: 28 > 13 > 6 > 3
    with tf.variable_scope(name):
        b1 = tf.layers.max_pooling2d(x, [3,3], 2, padding='SAME') # 96

        b2 = conv_bn_activ_dropout(x = x, n_filters = 96, kernel_size = [3,3], strides=2)

        b3 = conv_bn_activ_dropout(x = x, n_filters = 48, kernel_size = [1,1], strides=1)
        b3 = conv_bn_activ_dropout(x = b3, n_filters = 56, kernel_size = [3,3], strides=1)
        b3 = conv_bn_activ_dropout(x = b3, n_filters = 64, kernel_size = [3,3], strides=2)

        return tf.concat([b1, b2, b3], axis=-1)

def reduction_block_b(x, name='reduction_b'):
    # 256 => 384 (= 256 + 48 + 80)
    with tf.variable_scope(name):
        b1 = tf.layers.max_pooling2d(x, [3,3], 2, padding='SAME') # 256

        b2 = conv_bn_activ_dropout(x = x, n_filters = 48, kernel_size = [1,1], strides=1)
        b2 = conv_bn_activ_dropout(x = b2, n_filters = 48, kernel_size = [3,3], strides=2)

        b3 = conv_bn_activ_dropout(x = x, n_filters = 64, kernel_size = [1,1], strides=1)
        b3 = conv_bn_activ_dropout(x = b3, n_filters = 64, kernel_size = [1,7], strides=1)
        b3 = conv_bn_activ_dropout(x = b3, n_filters = 80, kernel_size = [7,1], strides=1)
        b3 = conv_bn_activ_dropout(x = b3, n_filters = 80, kernel_size = [3,3], strides=2)

        return tf.concat([b1, b2, b3], axis=-1)

# VGG19 모델을 정의합니다. 
def BUILD_INCEPTION_LIGHT(x):
    # 입력 이미지
    net = x
    n_filters = 64    
    dropout_rate = 0.7
    with tf.variable_scope('pre_inception'):
        b1 = conv_bn_activ_dropout(x = net, n_filters = 32, kernel_size = [1,1], strides=1)
        b1 = conv_bn_activ_dropout(x = b1, n_filters = 48, kernel_size = [3,3], strides=1)

        b2 = conv_bn_activ_dropout(x = net, n_filters = 32, kernel_size = [1,1], strides=1)
        b2 = conv_bn_activ_dropout(x = b2, n_filters = 32, kernel_size = [1,7], strides=1)
        b2 = conv_bn_activ_dropout(x = b2, n_filters = 32, kernel_size = [7,1], strides=1)
        b2 = conv_bn_activ_dropout(x = b2, n_filters = 48, kernel_size = [3,3], strides=1)

        net = tf.concat([b1, b2], axis=-1)
        assert net.shape[1:] == [32, 32, 96]
        
    with tf.variable_scope("inception-A"):
        for i in range(2):
            net = inception_block_a(net, name="inception-block-a{}".format(i))
        assert net.shape[1:] == [32, 32, 96]

    # reduction A
    # [28, 28, 96] => [14, 14, 256]
    with tf.variable_scope("reduction-A"):
        net = reduction_block_a(net)
        assert net.shape[1:] == [16, 16, 256]

    # inception B
    with tf.variable_scope("inception-B"):
        for i in range(3):
            net = inception_block_b(net, name="inception-block-b{}".format(i))
        assert net.shape[1:] == [16, 16, 256]

    # reduction B
    # [14, 14, 256] => [7, 7, 384]
    with tf.variable_scope("reduction-B"):
        net = reduction_block_b(net)
        assert net.shape[1:] == [8, 8, 384]

    # inception C
    with tf.variable_scope("inception-C"):
        for i in range(1):
            net = inception_block_c(net, name="inception-block-c{}".format(i))
        assert net.shape[1:] == [8, 8, 384]

    '''
    with tf.name_scope('Dense_Layer_01'):
        net = tf.contrib.layers.flatten(net)
        net = tf.layers.dense(net, 1024, activation=tf.nn.relu)
        net = tf.layers.dropout(net, 0.7, is_training)
    '''
    
    # GAP(Global Average Pooling) + dense
    with tf.variable_scope("fc"):
        net = tf.reduce_mean(net, [1,2]) # GAP
        assert net.shape[1:] == [384]
        net = tf.layers.dropout(net, 0.7, is_training)
                
    with tf.name_scope('Output_Layer'):
        logits = tf.layers.dense(net, 100, activation=None)
        y_pred = tf.nn.softmax(logits)

    return y_pred, logits

# 인풋 아웃풋 데이터, 드롭아웃 확률을 입력받기위한 플레이스홀더를 정의합니다.
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape=[None, 100])
keep_prob = tf.placeholder(tf.float32)

# CIFAR-10 데이터를 다운로드하고 데이터를 불러옵니다.
(X_train, Y_train), (X_test, Y_test) = load_data()

# scalar 형태의 레이블(0~9)을 One-hot Encoding 형태로 변환합니다.
Y_train_one_hot = tf.squeeze(tf.one_hot(Y_train, 100),axis=1)
Y_test_one_hot = tf.squeeze(tf.one_hot(Y_test, 100),axis=1)

# Convolutional Neural Networks(CNN) 그래프를 생성합니다.
y_pred, logits = BUILD_INCEPTION_LIGHT(x)

# Cross Entropy를 비용함수(loss function)으로 정의하고, RMSPropOptimizer를 이용해서 비용 함수를 최소화합니다.with tf.name_scope('optimizer'):

with tf.name_scope('Optimizer'):
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
    # Total_batch = int(X_train.shape[0]/batch_size)
    # Max Total_batch is 500
    Total_batch = 256
    # print("Size Train : ", X_train.shape[0])
    # print("Size Test  : ", X_test.shape[0])
    # print("Total batch : ", Total_batch)

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
        elapsed_time = datetime.timedelta(seconds=int(time.time()-start_time))
        print("[{}]".format(elapsed_time))
        
    print('Learning Finished!')

    # 최적화가 끝난 뒤, 변수를 저장합니다.
    #saver.save(sess, './model/dnn.ckpt', global_step=global_step)

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

