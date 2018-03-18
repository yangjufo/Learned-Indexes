from btree import BTree, Item
import pandas as pd
import tensorflow as tf
import numpy as np

LEARNING_RATE = 0.000001
TRAIN_STEPS = 20000
KEEP_RATIO = 0.9

BATCH_SIZE = 50
BLOCK_SIZE = 10
TOTAL_NUMBER = 10000

global test_x, test_y


class LearningIndexFramework:
    def __init__(self, index):
        return


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.random_normal(shape=shape)
    return tf.Variable(initial)


class TrainedNN:
    def __init__(self, in_num, hidden_num, out_num, x, y, next_stage_num):
        self.next_stage_num = next_stage_num
        self.in_num = in_num
        self.hidden_num = hidden_num
        self.out_num = out_num
        self.all_x = x
        self.all_y = []
        for i in y:
            self.all_y.append(i * next_stage_num / (TOTAL_NUMBER / BLOCK_SIZE))
        self.sess = tf.Session()
        self.max_abs_err = 0
        self.batch = 1
        self.train_x = np.array([self.all_x[0:BATCH_SIZE]]).T
        self.train_y = np.array([self.all_y[0:BATCH_SIZE]]).T

    def next_batch(self):
        if self.batch * BATCH_SIZE + BATCH_SIZE < len(self.all_x):
            self.train_x = np.array([self.all_x[self.batch * BATCH_SIZE:self.batch * BATCH_SIZE + BATCH_SIZE]]).T
            self.train_y = np.array([self.all_y[self.batch * BATCH_SIZE:self.batch * BATCH_SIZE + BATCH_SIZE]]).T
            self.batch += 1
        else:
            self.train_x = np.array([self.all_x[self.batch * BATCH_SIZE:len(self.all_x)]]).T
            self.train_y = np.array([self.all_y[self.batch * BATCH_SIZE:len(self.all_y)]]).T
            self.batch = 1

    def train(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.in_num])
        self.y_ = tf.placeholder(tf.float32, shape=[None, self.out_num])
        self.W_fc1 = weight_variable([self.in_num, self.hidden_num])
        self.b_fc1 = bias_variable([self.hidden_num])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.x, self.W_fc1) + self.b_fc1)
        # y = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

        self.keep_prob = tf.placeholder(tf.float32)
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

        self.W_fc3 = weight_variable([self.hidden_num, self.hidden_num])
        self.b_fc3 = bias_variable([self.hidden_num])
        self.h_fc3 = tf.nn.relu(tf.matmul(self.h_fc1_drop, self.W_fc3) + self.b_fc3)

        self.h_fc3_drop = tf.nn.dropout(self.h_fc3, self.keep_prob)

        self.W_fc2 = weight_variable([self.hidden_num, self.out_num])
        self.b_fc2 = bias_variable([self.out_num])
        self.y = tf.nn.relu(tf.matmul(self.h_fc3_drop, self.W_fc2) + self.b_fc2)

        self.cross_entropy = tf.reduce_mean(tf.pow((self.y - self.y_), 2))
        self.train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(self.cross_entropy)
        self.sess.run(tf.global_variables_initializer())

        for i in range(TRAIN_STEPS):
            if i % 1000 == 0:
                print "cross_entropy: %f" % self.sess.run(self.cross_entropy,
                                                          feed_dict={self.x: test_x, self.y_: test_y,
                                                                     self.keep_prob: 1.0})

                tmp_y = self.sess.run(self.y, feed_dict={self.x: self.train_x, self.keep_prob: KEEP_RATIO})
            self.sess.run(self.train_step,
                          feed_dict={self.x: self.train_x, self.y_: self.train_y, self.keep_prob: KEEP_RATIO})
            self.next_batch()

        result = self.sess.run(self.y, feed_dict={self.x: np.array([[97]]).T, self.keep_prob: 1.0})
        print result

    def predict(self, input):
        result = self.sess.run(self.y, feed_dict={self.x: np.array([[input]]).T, self.keep_prob: 1.0})
        return int(round(result[0]))

    def calc_err(self, input_data=None):
        if input_data is None:
            input_data = {}
        for k, v in input_data.iteritems():
            pos = self.predict(k)
            abs_err = abs(pos - v)
            if abs_err > self.max_abs_err:
                self.max_abs_err = abs_err


class TrainedBTree:
    def __init__(self, record=None):
        self.btree = BTree(2)
        if record is None:
            record = {}
        for k, v in record.iteritems():
            self.btree.insert(Item(k, v))


def hybrid_training(threshold, stages, recordData):
    M = len(stages)
    COL = 3
    tmp_records = [[{} for i in range(COL)] for i in range(M)]
    index = [[None for i in range(COL)] for i in range(M)]
    tmp_records[0][0] = recordData
    for i in range(0, M):
        for j in range(0, stages[i]):
            inputs = []
            labels = []
            for k, v in tmp_records[i][j].iteritems():
                inputs.append(k)
                labels.append(v)
            if i < M - 1:
                index[i][j] = TrainedNN(1, 32, 1, inputs, labels, stages[i + 1])
            else:
                index[i][j] = TrainedNN(1, 32, 1, inputs, labels, 1)
            index[i][j].train()
            if i < M - 1:
                for k, v in tmp_records[i][j].iteritems():
                    p = index[i][j].predict(k)
                    tmp_records[i + 1][p][k] = v

    for j in range(0, len(index[M - 1])):
        index[M - 1][j].calc_err(tmp_records[M - 1][j])
        if index[M - 1][j].max_abs_err > threshold:
            index[M - 1][j] = TrainedBTree(tmp_records[M - 1][j])
    return index


if __name__ == "__main__":
    path = "test_data.csv"
    data = pd.read_csv(path)
    train_set = {}
    test_set = {}
    tmp_x = []
    tmp_y = []
    for i in range(0, TOTAL_NUMBER):
        if i % 5 == 0:
            test_set[data.ix[i, 0]] = i / BLOCK_SIZE
            tmp_x.append(data.ix[i, 0])
            tmp_y.append(i / (TOTAL_NUMBER / 100))
        else:
            train_set[data.ix[i, 0]] = i / BLOCK_SIZE
    test_x = np.array([tmp_x]).T
    test_y = np.array([tmp_y]).T
    index = hybrid_training(8, [1, 100], train_set)
    print "yes"
