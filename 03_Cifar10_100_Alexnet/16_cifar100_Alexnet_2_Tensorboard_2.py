import tensorflow as tf
import numpy as np
import time, datetime
import os
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

DIR_Checkpoint  = "/tmp/ML/18_1_Cifar100_ALEXNET_2/CheckPoint"
DIR_Tensorboard = "/tmp/ML/18_1_Cifar100_ALEXNET_2/Tensorboard"

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
        
        tf.summary.histogram("x_image", x)
        tf.summary.histogram("Weights_01", weights['WEI_CONV_01'])
        
    with tf.name_scope('Conv_Layer_02'):
        # CONVOLUTION Layer
        CONV_02     = CONVOLUTION('CONV_02', NORM_01, weights['WEI_CONV_02'], biases['BIA_CONV_02'], padding='SAME')
        # Max Pooling (down-sampling)
        MAX_POOL_02 = POOLING('MAX_POOL_02',CONV_02,k=3, s=2, padding='SAME')
        # Apply Normalization
        NORM_02     = NORMALIZATION('NORM_02', MAX_POOL_02, lsize=5)

        tf.summary.histogram("Weights_02", weights['WEI_CONV_02'])
        
    with tf.name_scope('Conv_Layer_03'):
        # CONVOLUTION Layer
        CONV_03     = CONVOLUTION('CONV_03', NORM_02, weights['WEI_CONV_03'], biases['BIA_CONV_03'], padding='SAME')

        tf.summary.histogram("Weights_03", weights['WEI_CONV_03'])
        
    with tf.name_scope('Conv_Layer_04'):
        # CONVOLUTION Layer
        CONV_04     = CONVOLUTION('CONV_04', CONV_03, weights['WEI_CONV_04'], biases['BIA_CONV_04'], padding='SAME')

        tf.summary.histogram("Weights_04", weights['WEI_CONV_04'])
        
    with tf.name_scope('Conv_Layer_05'):
        # CONVOLUTION Layer
        CONV_05     = CONVOLUTION('CONV_05', CONV_04, weights['WEI_CONV_05'], biases['BIA_CONV_05'], padding='SAME')
        # Max Pooling (down-sampling)
        MAX_POOL_05 = POOLING('MAX_POOL_05',CONV_05,k=3, s=2, padding='SAME')
        # Apply Normalization
        NORM_05     = NORMALIZATION('NORM_05', MAX_POOL_05, lsize=5)

        tf.summary.histogram("Weights_05", weights['WEI_CONV_05'])
        
    with tf.name_scope('Dense_Layer_01'):
        # Fully connected layer
        DENSE_01=tf.reshape(NORM_05,[-1,weights['WEI_DENS_01'].get_shape().as_list()[0]])
        DENSE_01=tf.add(tf.matmul(DENSE_01,weights['WEI_DENS_01']),biases['BIA_DENS_01'])
        DENSE_01=tf.nn.relu(DENSE_01)

        ## dropout ##
        DENSE_01=tf.nn.dropout(DENSE_01, keep_prob)

        tf.summary.histogram("Weights_06", weights['WEI_DENS_01'])
        
    with tf.name_scope('Dense_Layer_02'):
        #### 2 fc ####
        #DENSE_02=tf.reshape(DENSE_01,[-1,weights['WEI_DENS_02'].get_shape().as_list()[0]])
        DENSE_02=tf.add(tf.matmul(DENSE_01,weights['WEI_DENS_02']),biases['BIA_DENS_02'])
        DENSE_02=tf.nn.relu(DENSE_02)

        ## dropout ##
        DENSE_02=tf.nn.dropout(DENSE_02, keep_prob)

        tf.summary.histogram("Weights_07", weights['WEI_DENS_02'])
        
    with tf.name_scope('Output_Layer'):
        #### output ####
        logits = tf.add(tf.matmul(DENSE_02,weights['out_w']),biases['out_b'])
        y_pred = tf.nn.softmax(logits)

        tf.summary.histogram("Weights_Out", weights['out_w'])
        tf.summary.histogram("Prediction", y_pred)
        
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
y_pred, logits = ALEXNET_2(x, weights, biases, keep_prob)

# Cross Entropy를 비용함수(loss function)으로 정의하고, RMSPropOptimizer를 이용해서 비용 함수를 최소화합니다.
with tf.name_scope('optimizer'):
    # Cross Entropy를 비용함수(loss function)으로 정의하고, RMSPropOptimizer를 이용해서 비용 함수를 최소화합니다.
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
    optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(loss, global_step=global_step)
    
    # tf.summary.scalar 를 사용하여 기록해야할 tensor들을 수집, tensor들이 너무 많으면 시각화가 복잡하므로 간단한것만 선택.
    tf.summary.scalar('cost', loss)
    
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
    
    # 텐서보드에서 표시해주기 위한 텐서들을 수집합니다.
    merged = tf.summary.merge_all()
    # 저장할 그래프와 텐서값들을 저장할 디렉토리를 설정합니다.
    writer = tf.summary.FileWriter(DIR_Tensorboard, sess.graph)
    # 이렇게 저장한 로그는, 학습 후 다음의 명령어를 이용해 웹서버를 실행시킨 뒤
    # tensorboard --logdir=./logs
    # 다음 주소와 웹브라우저를 이용해 텐서보드에서 확인할 수 있습니다.
    # http://localhost:6006

    # train my model
    print('Learning started. It takes sometime.')
    
    # train my model
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
        # print("Epoch: %6d, Loss: %2.6f" % (episode+1, total_cost/Total_batch))
        print('Global Step:', '%07d' % int(sess.run(global_step)/Total_batch), 'cost =', '{:02.6f}'.format(total_cost/Total_batch))
        
        elapsed_time = datetime.timedelta(seconds=int(time.time()-start_time))
        print("[{}]".format(elapsed_time))

        # 적절한 시점에 저장할 값들을 수집하고 저장합니다.
        summary = sess.run(merged, feed_dict={x: batch[0], y: batch[1], keep_prob: 0.8})
        writer.add_summary(summary, global_step=sess.run(global_step))

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