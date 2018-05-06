import tensorflow as tf
import numpy as np
from enum import Enum
from data.create_data import Distribution


DATA_TYPE = Distribution.RANDOM


def set_data_type(data_type):
    global DATA_TYPE
    DATA_TYPE = data_type


class Parameter:
    def __init__(self, stages, cores, train_steps, batch_sizes, learning_rates, keep_ratios):
        self.stage_set = stages
        self.core_set = cores
        self.train_step_set = train_steps
        self.batch_size_set = batch_sizes
        self.learning_rate_set = learning_rates
        self.keep_ratio_set = keep_ratios


class ParameterPool(Enum):
    RANDOM = Parameter(stages=[1, 10], cores=[[1, 1], [1, 1]], train_steps=[20000, 2000],
                       batch_sizes=[50, 50], learning_rates=[0.0001, 0.001], keep_ratios=[0.9, 0.9])
    LOGNORMAL = Parameter(stages=[1, 100], cores=[[1, 16, 16, 1], [1, 8, 1]], train_steps=[2000, 400],
                          batch_sizes=[100, 50], learning_rates=[0.0001, 0.001], keep_ratios=[1.0, 0.9])
    EXPONENTIAL = Parameter(stages=[1, 100], cores=[[1, 8, 1], [1, 1]], train_steps=[1700, 2000],
                            batch_sizes=[50, 50], learning_rates=[0.001, 0.001], keep_ratios=[0.5, 1.0])
    # EXPONENTIAL = Parameter(stages=[1, 100], cores=[[1, 16, 16, 1], [1, 8, 1]], train_steps=[20000, 300],
    #                       batch_sizes=[20, 50], learning_rates=[0.0001, 0.001], keep_ratios=[1.0, 1.0])
    NORMAL = Parameter(stages=[1, 100], cores=[[1, 8, 1], [1, 8, 1]], train_steps=[20000, 300],
                       batch_sizes=[50, 50], learning_rates=[0.0001, 0.001], keep_ratios=[0.9, 1.0])


def weight_variable(shape):
    if DATA_TYPE == Distribution.RANDOM:
        initial = tf.constant(0.1, shape=shape)
    elif DATA_TYPE == Distribution.LOGNORMAL:
        initial = tf.truncated_normal(shape=shape, stddev=0.1)
        # initial = tf.constant(0.1, shape=shape)
    elif DATA_TYPE == Distribution.EXPONENTIAL:
        # initial = tf.truncated_normal(shape=shape, stddev=0.1)
        initial = tf.constant(0.1, shape=shape)
    elif DATA_TYPE == Distribution.NORMAL:
        initial = tf.truncated_normal(shape=shape, mean=0.1, stddev=0.1)
    else:
        initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


class TrainedNN:
    def __init__(self, cores, train_step_num, batch_size, learning_rate, keep_ratio, train_x, train_y, test_x, test_y):
        if cores is None:
            cores = []
        self.core_nums = cores
        self.train_step_nums = train_step_num
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.keep_ratio = keep_ratio
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = np.array([test_x]).T
        self.test_y = np.array([test_y]).T
        self.sess = tf.Session()
        self.batch = 1
        self.batch_x = np.array([self.train_x[0:self.batch_size]]).T
        self.batch_y = np.array([self.train_y[0:self.batch_size]]).T
        self.y_ = tf.placeholder(tf.float32, shape=[None, self.core_nums[-1]])
        self.w_fc = []
        self.b_fc = []
        for i in range(len(self.core_nums) - 1):
            self.w_fc.append(weight_variable([self.core_nums[i], self.core_nums[i + 1]]))
            self.b_fc.append(bias_variable([self.core_nums[i + 1]]))
        self.h_fc = [None for i in range(len(self.core_nums))]
        self.h_fc_drop = [None for i in range(len(self.core_nums))]
        self.h_fc_drop[0] = tf.placeholder(tf.float32, shape=[None, self.core_nums[0]])
        self.keep_prob = tf.placeholder(tf.float32)

    def next_batch(self):
        if self.batch * self.batch_size + self.batch_size < len(self.train_x):
            self.batch_x = np.array([self.train_x[self.batch * self.batch_size:(self.batch + 1) * self.batch_size]]).T
            self.batch_y = np.array([self.train_y[self.batch * self.batch_size:(self.batch + 1) * self.batch_size]]).T
            self.batch += 1
        else:
            self.batch_x = np.array([self.train_x[self.batch * self.batch_size:len(self.train_x)]]).T
            self.batch_y = np.array([self.train_y[self.batch * self.batch_size:len(self.train_y)]]).T
            self.batch = 0

    def train(self):
        for i in range(len(self.core_nums) - 1):
            self.h_fc[i] = tf.nn.relu(tf.matmul(self.h_fc_drop[i], self.w_fc[i]) + self.b_fc[i])
            self.h_fc_drop[i + 1] = tf.nn.dropout(self.h_fc[i], self.keep_prob)

        self.cross_entropy = tf.reduce_mean(tf.losses.mean_squared_error(self.y_, self.h_fc[len(self.core_nums) - 2]))
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy)
        self.sess.run(tf.global_variables_initializer())

        for step in range(0, self.train_step_nums):
            self.sess.run(self.train_step,
                          feed_dict={self.h_fc_drop[0]: self.batch_x, self.y_: self.batch_y,
                                     self.keep_prob: self.keep_ratio})

            tmp_w0 = self.sess.run(self.w_fc[0], feed_dict={self.h_fc_drop[0]: self.batch_x,
                                                             self.y_: self.batch_y,
                                                             self.keep_prob: 1.0})
            # tmp_w1 = self.sess.run(self.w_fc[1], feed_dict={self.h_fc_drop[0]: self.batch_x,
            #                                                 self.y_: self.batch_y,
            #                                                 self.keep_prob: 1.0})
            # tmp_w2 = self.sess.run(self.w_fc[2], feed_dict={self.h_fc_drop[0]: self.batch_x,
            #                                                 self.y_: self.batch_y,
            #                                                 self.keep_prob: 1.0})
            # tmp_c = self.sess.run(self.cross_entropy, feed_dict={self.h_fc_drop[0]: self.batch_x,
            #                                                      self.y_: self.batch_y,
            #                                                      self.keep_prob: 1.0})
            tmp_y = self.sess.run(self.h_fc[0], feed_dict={self.h_fc_drop[0]: self.batch_x,
                                                            self.y_: self.batch_y,
                                                            self.keep_prob: 1.0})

            if step != 0 and step % 100 == 0:
                err = self.sess.run(self.cross_entropy, feed_dict={self.h_fc_drop[0]: np.array([self.train_x]).T,
                                                                   self.y_: np.array([self.train_y]).T,
                                                                   self.keep_prob: 1.0})
                print("cross_entropy: %f" % err)
                if err < 1:
                    return

            self.next_batch()

    def predict(self, input_key):
        result_pos = self.sess.run(self.h_fc[len(self.core_nums) - 2],
                                   feed_dict={self.h_fc_drop[0]: np.array([[input_key]]).T, self.keep_prob: 1.0})
        return int(round(result_pos[0]))

    def cal_err(self):
        mean_err = self.sess.run(self.cross_entropy, feed_dict={self.h_fc_drop[0]: np.array([self.train_x]).T,
                                                                self.y_: np.array([self.train_y]).T,
                                                                self.keep_prob: 1.0})
        return mean_err

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def get_weight(self):
        weight = []
        for i in range(len(self.core_nums) - 1):
            weight.append(self.sess.run(self.w_fc[i], feed_dict={self.h_fc_drop[0]: np.array([self.train_x]).T,
                                                                self.y_: np.array([self.train_y]).T,
                                                                self.keep_prob: 1.0}).tolist())
        return weight

    def get_bias(self):
        bias = []
        for i in range(len(self.core_nums) - 1):
            bias.append(self.sess.run(self.b_fc[i], feed_dict={self.h_fc_drop[0]: np.array([self.train_x]).T,
                                                                 self.y_: np.array([self.train_y]).T,
                                                                 self.keep_prob: 1.0}).tolist())
        return bias
