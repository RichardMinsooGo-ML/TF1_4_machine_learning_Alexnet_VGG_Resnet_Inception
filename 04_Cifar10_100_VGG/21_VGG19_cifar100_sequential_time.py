import tensorflow as tf
import numpy as np
import time, datetime
# CIFAR-10 데이터를 다운로드 받기 위한 keras의 helper 함수인 load_data 함수를 임포트합니다.
from tensorflow.keras.datasets.cifar100 import load_data

N_EPISODES = 5
batch_size = 100
keep_prob = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)
OUTPUT_SIZE = 100

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

# CNN 모델을 정의합니다. 
def BUILD_NETWORK_CNN(x):
    # 입력 이미지
    net = x
    
    with tf.name_scope('Conv_Layer_01_01'):
        # 첫번째 convolutional layer - 하나의 grayscale 이미지를 64개의 특징들(feature)으로 맵핑(maping).     
        net = tf.layers.conv2d(net, 64, [3, 3], activation=tf.nn.relu, padding='SAME')
        net = tf.layers.batch_normalization(net)
        net = tf.layers.dropout(net, 0.7, is_training)
    
    with tf.name_scope('Conv_Layer_01_02'):
        # 첫번째 convolutional layer - 하나의 grayscale 이미지를 64개의 특징들(feature)으로 맵핑(maping).     
        net = tf.layers.conv2d(net, 64, [3, 3], activation=tf.nn.relu, padding='SAME')
        net = tf.layers.batch_normalization(net)
        net = tf.layers.dropout(net, 0.7, is_training)
    
    with tf.name_scope('Pool_Layer_01'):
        # 첫번째 Pooling layer
        h_pool1 = tf.layers.max_pooling2d(net, [2, 2], [2, 2], padding='SAME')

    with tf.name_scope('Conv_Layer_02_01'):
        # 첫번째 convolutional layer - 하나의 grayscale 이미지를 64개의 특징들(feature)으로 맵핑(maping).     
        net = tf.layers.conv2d(net, 128, [3, 3], activation=tf.nn.relu, padding='SAME')
        net = tf.layers.batch_normalization(net)
        net = tf.layers.dropout(net, 0.7, is_training)
    
    with tf.name_scope('Conv_Layer_02_02'):
        # 첫번째 convolutional layer - 하나의 grayscale 이미지를 64개의 특징들(feature)으로 맵핑(maping).     
        net = tf.layers.conv2d(net, 128, [3, 3], activation=tf.nn.relu, padding='SAME')
        net = tf.layers.batch_normalization(net)
        net = tf.layers.dropout(net, 0.7, is_training)
    
    with tf.name_scope('Pool_Layer_02'):
        # 첫번째 Pooling layer
        net = tf.layers.max_pooling2d(net, [2, 2], [2, 2], padding='SAME')
        
    with tf.name_scope('Conv_Layer_03_01'):
        # 첫번째 convolutional layer - 하나의 grayscale 이미지를 64개의 특징들(feature)으로 맵핑(maping).     
        net = tf.layers.conv2d(net, 256, [3, 3], activation=tf.nn.relu, padding='SAME')
        net = tf.layers.batch_normalization(net)
        net = tf.layers.dropout(net, 0.7, is_training)
    
    with tf.name_scope('Conv_Layer_03_02'):
        # 첫번째 convolutional layer - 하나의 grayscale 이미지를 64개의 특징들(feature)으로 맵핑(maping).     
        net = tf.layers.conv2d(net, 256, [3, 3], activation=tf.nn.relu, padding='SAME')
        net = tf.layers.batch_normalization(net)
        net = tf.layers.dropout(net, 0.7, is_training)
    
    with tf.name_scope('Conv_Layer_03_03'):
        # 첫번째 convolutional layer - 하나의 grayscale 이미지를 64개의 특징들(feature)으로 맵핑(maping).     
        net = tf.layers.conv2d(net, 256, [3, 3], activation=tf.nn.relu, padding='SAME')
        net = tf.layers.batch_normalization(net)
        net = tf.layers.dropout(net, 0.7, is_training)
    
    with tf.name_scope('Pool_Layer_03'):
        # 첫번째 Pooling layer
        net = tf.layers.max_pooling2d(net, [2, 2], [2, 2], padding='SAME')
        
    with tf.name_scope('Conv_Layer_04_01'):
        # 첫번째 convolutional layer - 하나의 grayscale 이미지를 64개의 특징들(feature)으로 맵핑(maping).     
        net = tf.layers.conv2d(net, 512, [3, 3], activation=tf.nn.relu, padding='SAME')
        net = tf.layers.batch_normalization(net)
        net = tf.layers.dropout(net, 0.7, is_training)
    
    with tf.name_scope('Conv_Layer_04_02'):
        # 첫번째 convolutional layer - 하나의 grayscale 이미지를 64개의 특징들(feature)으로 맵핑(maping).     
        net = tf.layers.conv2d(net, 512, [3, 3], activation=tf.nn.relu, padding='SAME')
        net = tf.layers.batch_normalization(net)
        net = tf.layers.dropout(net, 0.7, is_training)
    
    with tf.name_scope('Conv_Layer_04_03'):
        # 첫번째 convolutional layer - 하나의 grayscale 이미지를 64개의 특징들(feature)으로 맵핑(maping).     
        net = tf.layers.conv2d(net, 512, [3, 3], activation=tf.nn.relu, padding='SAME')
        net = tf.layers.batch_normalization(net)
        net = tf.layers.dropout(net, 0.7, is_training)
    
    with tf.name_scope('Pool_Layer_04'):
        # 첫번째 Pooling layer
        net = tf.layers.max_pooling2d(net, [2, 2], [2, 2], padding='SAME')
        
    with tf.name_scope('Conv_Layer_05_01'):
        # 첫번째 convolutional layer - 하나의 grayscale 이미지를 64개의 특징들(feature)으로 맵핑(maping).     
        net = tf.layers.conv2d(net, 512, [3, 3], activation=tf.nn.relu, padding='SAME')
        net = tf.layers.batch_normalization(net)
        net = tf.layers.dropout(net, 0.7, is_training)
    
    with tf.name_scope('Conv_Layer_05_02'):
        # 첫번째 convolutional layer - 하나의 grayscale 이미지를 64개의 특징들(feature)으로 맵핑(maping).     
        net = tf.layers.conv2d(net, 512, [3, 3], activation=tf.nn.relu, padding='SAME')
        net = tf.layers.batch_normalization(net)
        net = tf.layers.dropout(net, 0.7, is_training)
    
    with tf.name_scope('Conv_Layer_05_03'):
        # 첫번째 convolutional layer - 하나의 grayscale 이미지를 64개의 특징들(feature)으로 맵핑(maping).     
        net = tf.layers.conv2d(net, 512, [3, 3], activation=tf.nn.relu, padding='SAME')
        net = tf.layers.batch_normalization(net)
        net = tf.layers.dropout(net, 0.7, is_training)
    
    with tf.name_scope('Pool_Layer_05'):
        # 첫번째 Pooling layer
        net = tf.layers.max_pooling2d(net, [2, 2], [2, 2], padding='SAME')
        
    with tf.name_scope('Dense_Layer_01'):
        net = tf.contrib.layers.flatten(net)
        net = tf.layers.dense(net, 4096, activation=tf.nn.relu)
        net = tf.layers.dropout(net, 0.7, is_training)

    with tf.name_scope('Dense_Layer_02'):
        net = tf.layers.dense(net, 4096, activation=tf.nn.relu)
        net = tf.layers.dropout(net, 0.7, is_training)

    with tf.name_scope('Output_Layer'):
        # Fully Connected Layer 2 - 384개의 특징들(feature)을 10개의 클래스-airplane, automobile, bird...-로 맵핑(maping)합니다.
        logits = tf.layers.dense(net, OUTPUT_SIZE, activation=None)
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
y_pred, logits = BUILD_NETWORK_CNN(x)

# Cross Entropy를 비용함수(loss function)으로 정의하고, RMSPropOptimizer를 이용해서 비용 함수를 최소화합니다.with tf.name_scope('optimizer'):

with tf.name_scope('Optimizer'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
    optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

# 정확도를 계산하는 연산을 추가합니다.
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 세션을 열어 실제 학습을 진행합니다.
with tf.Session() as sess:
    
    # 모든 변수들을 초기화한다. 
    sess.run(tf.global_variables_initializer())
    
    start_time = time.time()
    # train my model
    print('Learning started. It takes sometime.')

    # train my model
    # Total_batch = int(X_train.shape[0]/batch_size)
    Total_batch = 32
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

    # 학습이 끝나면 테스트 데이터(10000개)에 대한 정확도를 출력합니다.  
    test_accuracy = 0.0  
    for i in range(3):
        test_batch = next_batch(1000, X_test, Y_test_one_hot.eval())
        test_accuracy = test_accuracy + accuracy.eval(feed_dict={x: test_batch[0], y: test_batch[1], keep_prob: 1.0})
    test_accuracy = test_accuracy / 3;
    print("Test Data Accuracy: %2.4f" % test_accuracy)
    elapsed_time = time.time() - start_time
    formatted = datetime.timedelta(seconds=int(elapsed_time))
    print("=== training time elapsed: {}s ===".format(formatted))

