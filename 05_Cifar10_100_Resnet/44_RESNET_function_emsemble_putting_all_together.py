import tensorflow as tf
import numpy as np
import time, datetime
import os
import matplotlib.pyplot as plt
import random
tf.set_random_seed(777)  # reproducibility
# 최신 Windows Laptop에서만 사용할것.CPU Version이 높을때 사용.
# AVX를 지원하는 CPU는 Giuthub: How to compile tensorflow using SSE4.1, SSE4.2, and AVX. 
# Ubuntu와 MacOS는 지원하지만 Windows는 없었음. 2018-09-29
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Compuntational Graph Initialization
from tensorflow.python.framework import ops
ops.reset_default_graph()

# CIFAR-10 데이터를 다운로드 받기 위한 keras의 helper 함수인 load_data 함수를 임포트합니다.
from tensorflow.keras.datasets.cifar10 import load_data

# CIFAR-100 데이터를 다운로드 받기 위한 keras의 helper 함수인 load_data 함수를 임포트합니다.
# from tensorflow.keras.datasets.cifar100 import load_data

# hyper parameters
Alpha_Lr   = 0.001
N_EPISODES = 10
# batch size, 25, 50, 100, 200, 500, 1000, 2000, 5000
training_batch_size = 200
test_batch_size = 1000

# dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
keep_prob = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)

# for cifar 10
N_Classes = 10

# for cifar 100
# N_Classes = 100

# CIFAR-10 데이터를 다운로드하고 데이터를 불러옵니다.
(X_train, Y_train), (X_test, Y_test) = load_data()

# scalar 형태의 레이블(0~9)을 One-hot Encoding 형태로 변환합니다.
Y_train_one_hot = tf.squeeze(tf.one_hot(Y_train, N_Classes),axis=1)
Y_test_one_hot = tf.squeeze(tf.one_hot(Y_test, N_Classes),axis=1)

DIR_Checkpoint  = "/tmp/ML/Working_Folder_5_RESNET/CheckPoint"
DIR_Tensorboard = "/tmp/ML/Working_Folder_5_RESNET/Tensorboard"

# 학습에 직접적으로 사용하지 않고 학습 횟수에 따라 단순히 증가시킬 변수를 만듭니다.
global_step = tf.Variable(0, trainable=False, name='global_step')

# 다음 배치를 읽어오기 위한 next_batch 유틸리티 함수를 정의합니다.
def next_batch_random(num, data, labels):
    '''
    `num` 개수 만큼의 랜덤한 샘플들과 레이블들을 리턴합니다.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

# 다음 배치를 읽어오기 위한 Next_batch_sequential 유틸리티 함수를 정의합니다.
def Next_batch_sequential(index, num, data, labels):
    # index는 시작 번호 이고, num만큼 데이터를 순서대로 부릅니다.
    idx = np.arange(0 , len(data))
    idx = idx[index:index + num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

    
def RESIDUAL_BLOCK( x, output_channel, downsampling=False, name='res_block'):
    input_channel = int(x.shape[-1]) # get # of input channels

    if downsampling:
        stride = 2
    else:
        stride = 1

    with tf.variable_scope(name):
        with tf.variable_scope('conv1_in_block'):
            # h1 = tf.layers.conv2d(x, output_channel, [3,3], strides=stride, padding='SAME', use_bias=False,
            #                       kernel_initializer=tf.contrib.layers.variance_scaling_initializer(seed=seed))
            h1 = tf.layers.conv2d(x, output_channel, [3,3], strides=stride, padding='SAME')
            h1 = tf.layers.batch_normalization(h1)
            h1 = tf.nn.relu(h1)

        with tf.variable_scope('conv2_in_block'):
            # h2 = tf.layers.conv2d(h1, output_channel, [3,3], strides=1, padding='SAME', use_bias=False,
            #                       kernel_initializer=tf.contrib.layers.variance_scaling_initializer(seed=seed))
            h2 = tf.layers.conv2d(h1, output_channel, [3,3], strides=1, padding='SAME')
            h2 = tf.layers.batch_normalization(h2)

        # option A - zero padding for extra dimension => no need extra params
        # if downsampling:
        #     pooled_x = tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding='valid')
        #     padded_x = tf.pad(pooled_x, [[0,0], [0,0], [0,0], [input_channel // 2, input_channel // 2]])
        # else:
        #     padded_x = x

        # option B - projection with 1x1 strided conv
        if downsampling:
            # x = tf.layers.conv2d(x, output_channel, [1,1], strides=stride, padding='SAME', 
            #     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(seed=seed))
            x = tf.layers.conv2d(x, output_channel, [1,1], strides=stride, padding='SAME')
        return tf.nn.relu(h2 + x)

# CNN 모델을 정의합니다. 
class CL_Deep_CNN:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self.FN_Build_Network()

    def FN_Build_Network(self):
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            self.keep_prob = tf.placeholder(tf.float32)

            # input place holders
            self.X = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
            self.Y = tf.placeholder(tf.float32, shape=[None, N_Classes])
            net = self.X
            n_filters = 64
            #"6n+2: {3, 5, 7, 9 ... 18} (default: 3)"
            layer_n = 3
            with tf.variable_scope("conv0"):
                # net = tf.layers.conv2d(net, 16, [3,3], strides=1, padding='SAME', use_bias=False,
                #           kernel_initializer=tf.contrib.layers.variance_scaling_initializer(seed=seed))
                net = tf.layers.conv2d(net, 16, [3,3], strides=1, activation=tf.nn.relu, padding='SAME')
                net = tf.layers.batch_normalization(net)
                # net = tf.nn.relu(net)

            with tf.variable_scope("conv1"):
                for i in range(layer_n):
                    net = RESIDUAL_BLOCK(x = net, output_channel = 16, name="resblock{}".format(i+1))
                    assert net.shape[1:] == [32, 32, 16]

            with tf.variable_scope("conv2"):
                for i in range(layer_n):
                    net = RESIDUAL_BLOCK(x = net, output_channel = 32, downsampling=(i==0), name="resblock{}".format(i+1))
                    assert net.shape[1:] == [16, 16, 32]

            with tf.variable_scope("conv3"):
                for i in range(layer_n):
                    net = RESIDUAL_BLOCK(x = net, output_channel = 64, downsampling=(i==0), name="resblock{}".format(i+1))
                    assert net.shape[1:] == [8, 8, 64]

            # GAP(Global Average Pooling) + dense
            with tf.variable_scope("fc"):
                net = tf.reduce_mean(net, [1,2]) # GAP
                assert net.shape[1:] == [64]
                # net = tf.layers.dropout(net, 0.7, is_training)

            with tf.name_scope('Output_Layer'):
                # Fully Connected Layer 2 - 384개의 특징들(feature)을 10개의 클래스-airplane, automobile, bird...-로 맵핑(maping)합니다.
                logits = tf.layers.dense(net, N_Classes, activation=None)
                self.y_pred = tf.nn.softmax(logits)
                       
        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate = Alpha_Lr).minimize(self.cost, global_step=global_step)

        correct_prediction = tf.equal(tf.argmax(self.y_pred, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def FN_Predict(self, x_test, keep_prob=1.0):
        return self.sess.run(self.y_pred, feed_dict={self.X: x_test, self.keep_prob: keep_prob})

    def FN_Get_Accuracy(self, x_test, y_test, keep_prob=1.0):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: keep_prob})

    def FN_Train_Net(self, x_data, y_data, keep_prob=0.7):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.keep_prob: keep_prob})

# 세션을 열어 실제 학습을 진행합니다.
with tf.Session() as sess:
    
    models = []
    num_models = 3
    for m in range(num_models):
        models.append(CL_Deep_CNN(sess, "model" + str(m)))
    
    # 모든 변수들을 초기화한다.  
    init = tf.global_variables_initializer()
    sess.run(init)
    
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
    # Total_batch = int(X_train.shape[0]/training_batch_size)
    Total_batch = 20

    episode = 0
    cycle_time = 540
    target_accuracy = 0
    kkk = 0
    while (kkk < 3) and (target_accuracy < 0.8):
        while time.time() < start_time + cycle_time*(kkk + 1):
        # for episode in range(N_EPISODES):
            avg_cost_list = np.zeros(len(models))
            for i in range(Total_batch):
                index = i*training_batch_size
                batch = Next_batch_sequential(index,training_batch_size, X_train, Y_train_one_hot.eval())

                # c, _ = m1.FN_Train_Net(batch[0], batch[1])
                # train each model
                for m_idx, m in enumerate(models):
                    c, _ = m.FN_Train_Net(batch[0], batch[1])
                    avg_cost_list[m_idx] += c / Total_batch
                    # avg_cost += c / Total_batch

            print('Global Step:', '%05d' % int(sess.run(global_step)/Total_batch/num_models), 'cost =',avg_cost_list)

            elapsed_time = datetime.timedelta(seconds=int(time.time()-start_time))
            now = datetime.datetime.now()
            print('Elapsed : ',"[{}]".format(elapsed_time),'Now : ',now)
            episode += 1

        print('Learning Finished!')
        saver.save(sess, DIR_Checkpoint + './dnn.ckpt', global_step=global_step)


        # Test model and check accuracy
        test_size = int(X_test.shape[0])
        N_test_batch = int(X_test.shape[0]/test_batch_size)
        predictions = np.zeros([test_size, N_Classes])
        # print("predictions shape = ",np.shape(predictions))
        # print("Y_test_one_hot.eval() shape = ",np.shape(Y_test_one_hot.eval()))

        for m_idx, m in enumerate(models):
            test_accuracy = 0.0
            p = np.zeros([test_batch_size, N_Classes])

            for i in range(N_test_batch):
                index = i*test_batch_size
                test_batch = Next_batch_sequential(index, test_batch_size, X_test, Y_test_one_hot.eval())
                test_accuracy = test_accuracy + m.FN_Get_Accuracy(test_batch[0], test_batch[1])
                Pred_per_batch = m.FN_Predict(test_batch[0])
                # print("p shape = ",np.shape(p))
                # print("Pred_per_batch shape = ",np.shape(Pred_per_batch))
                if i == 0:
                    p += Pred_per_batch
                    # print(i, "p shape = ",np.shape(p))
                if i > 0:
                    p = np.concatenate([p,Pred_per_batch], axis = 0)
                    # print(i, "p shape = ",np.shape(p))

            test_accuracy = test_accuracy / N_test_batch

            print("Ensemble Model ",m_idx + 1, "test data Accuracy: %2.4f" % test_accuracy)
            # print(m_idx, 'Accuracy:', m.FN_Get_Accuracy(test_batch[0], test_batch[1]))

            predictions += p

        ensemble_is_correct = tf.equal(tf.argmax(predictions, 1), tf.argmax(Y_test_one_hot.eval(), 1))
        ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_is_correct, tf.float32))
        target_accuracy = sess.run(ensemble_accuracy)
        print('Ensemble accuracy:', target_accuracy)
        
        elapsed_time = time.time() - start_time
        formatted = datetime.timedelta(seconds=int(elapsed_time))
        print("=== training time elapsed: {}s ===".format(formatted))
        kkk += 1
        
    
    # 결과 확인 (matplot) for cifar 10
    # Option 1 : sequential plotting
    # index = 20
    # test_batch = Next_batch_sequential(index, 100, X_test, Y_test_one_hot.eval())
    
    # Option 2 : random plotting
    test_batch = next_batch_random(100, X_test, Y_test_one_hot.eval())
    labels = m.FN_Predict(test_batch[0],1)
    plot_test = test_batch[0]
    CIFAR10_LABELS_LIST = [
        'airplane', 
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck'
    ]
    fig = plt.figure()
    
    for i in range(60):
        subplot = fig.add_subplot(4, 15, i + 1)
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.set_title('%s' % CIFAR10_LABELS_LIST[np.argmax(labels[i])])
        subplot.imshow(plot_test[i])

    plt.show()
    
    
    """
    # 결과 확인 (matplot) for cifar 100
    # Option 1 : sequential plotting
    # index = 20
    # test_batch = Next_batch_sequential(index, 100, X_test, Y_test_one_hot.eval())
    
    # Option 2 : random plotting
    test_batch = next_batch_random(100, X_test, Y_test_one_hot.eval())
    labels = m.FN_Predict(test_batch[0],1)
    plot_test = test_batch[0]
    CIFAR100_LABELS_LIST = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm'
    ]
    fig = plt.figure()
    
    for i in range(60):
        subplot = fig.add_subplot(4, 15, i + 1)
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.set_title('%s' % CIFAR100_LABELS_LIST[np.argmax(labels[i])])
        subplot.imshow(plot_test[i])

    plt.show()
    """    

