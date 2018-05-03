from __future__ import print_function
import pandas as pd
from Trained_NN import TrainedNN, ParameterPool, Distribution, set_data_type
from btree import BTree
import time
import gc

BLOCK_SIZE = 100
TOTAL_NUMBER = 1000000

filePath = {
    Distribution.RANDOM: "data/random.csv",
    Distribution.BINOMIAL: "data/binomial.csv",
    Distribution.POISSON: "data/poisson.csv",
    Distribution.EXPONENTIAL: "data/exponential.csv",
    Distribution.NORMAL: "data/normal.csv",
    Distribution.LOGNORMAL: "data/lognormal.csv"
}


def hybrid_training(stage_nums, core_nums, train_step_nums, batch_size_nums, learning_rate_nums, keep_ratio_nums,
                    train_data_x, train_data_y, test_data_x, test_data_y):
    stage_length = len(stage_nums)
    col_num = stage_nums[1]
    tmp_inputs = [[[] for i in range(col_num)] for i in range(stage_length)]
    tmp_labels = [[[] for i in range(col_num)] for i in range(stage_length)]
    index = [[None for i in range(col_num)] for i in range(stage_length)]
    tmp_inputs[0][0] = train_data_x
    tmp_labels[0][0] = train_data_y
    test_inputs = test_data_x
    for i in range(0, stage_length):
        for j in range(0, stage_nums[i]):
            if len(tmp_labels[i][j]) == 0:
                continue
            inputs = tmp_inputs[i][j]
            labels = []
            test_labels = []
            if i == 0:
                divisor = stage_nums[i + 1] * 1.0 / (TOTAL_NUMBER / BLOCK_SIZE)
                for k in tmp_labels[i][j]:
                    labels.append(int(k * divisor))
                for k in test_data_y:
                    test_labels.append(int(k * divisor))
            else:
                labels = tmp_labels[i][j]
                test_labels = test_data_y
            index[i][j] = TrainedNN(core_nums[i], train_step_nums[i], batch_size_nums[i], learning_rate_nums[i],
                                    keep_ratio_nums[i], inputs, labels, test_inputs, test_labels)
            index[i][j].train()

            if i < stage_length - 1:
                for ind in range(len(tmp_inputs[i][j])):
                    p = index[i][j].predict(tmp_inputs[i][j][ind])
                    if p > stage_nums[i + 1] - 1:
                        p = stage_nums[i + 1] - 1
                    tmp_inputs[i + 1][p].append(tmp_inputs[i][j][ind])
                    tmp_labels[i + 1][p].append(tmp_labels[i][j][ind])
                # index[i][j].save("model/" + str(i) + "-" + str(j))
                # del index[i][j]
                # gc.collect()
    for i in range(stage_nums[stage_length - 1]):
        if index[stage_length - 1][i] is None:
            continue
        mean_abs_err = index[stage_length - 1][i].cal_err()
        if mean_abs_err > 100:
            index[stage_length - 1][i] = BTree(2)
            index[stage_length - 1][i].build(tmp_inputs[stage_length - 1][i], tmp_labels[stage_length - 1][i])
        # else:
        #     index[stage_length - 1][i].save("model/" + str(i) + "-" + str(j))
        #     del [stage_length - 1][i]
        #     gc.collect()
    return index


def train_index(distribution):
    path = filePath[distribution]
    data = pd.read_csv(path)
    train_set_x = []
    train_set_y = []
    test_set_x = []
    test_set_y = []

    set_data_type(distribution)
    if distribution == Distribution.RANDOM:
        parameter = ParameterPool.RANDOM.value
    elif distribution == Distribution.LOGNORMAL:
        parameter = ParameterPool.LOGNORMAL.value
    elif distribution == Distribution.EXPONENTIAL:
        parameter = ParameterPool.EXPONENTIAL.value
    elif distribution == Distribution.NORMAL:
        parameter = ParameterPool.NORMAL.value
    else:
        return
    stage_set = parameter.stage_set
    core_set = parameter.core_set
    train_step_set = parameter.train_step_set
    batch_size_set = parameter.batch_size_set
    learning_rate_set = parameter.learning_rate_set
    keep_ratio_set = parameter.keep_ratio_set

    start = 0

    for i in range(start, start + TOTAL_NUMBER - 1):
        if i % 10 == 0:
            test_set_x.append(data.ix[i, 0])
            test_set_y.append(data.ix[i, 1])
        train_set_x.append(data.ix[i, 0])
        train_set_y.append(data.ix[i, 1])

    print("*************strat Learned NN************")
    print("Start Train")
    start_time = time.time()
    trained_index = hybrid_training(stage_set, core_set, train_step_set, batch_size_set, learning_rate_set,
                                    keep_ratio_set, train_set_x, train_set_y, test_set_x, test_set_y)
    end_time = time.time()
    print("Build Learned NN time ", end_time - start_time)
    print("Calculate Error")
    err = 0
    start_time = time.time()
    for ind in range(len(test_set_x)):
        pre1 = trained_index[0][0].predict(test_set_x[ind])
        if pre1 > stage_set[1] - 1:
            pre1 = stage_set[1] - 1
        pre2 = trained_index[1][pre1].predict(test_set_x[ind])
        err += abs(pre2 - test_set_y[ind])
    end_time = time.time()
    print("Search time ", (end_time - start_time) / len(test_set_x))
    print("mean error = ", err * 1.0 / len(test_set_x))
    print("*************end Learned NN************\n\n")
    del trained_index
    gc.collect()
    print("*************start BTree************")
    bt = BTree(2)
    print("Start Build")
    start_time = time.time()
    bt.build(train_set_x, train_set_y)
    end_time = time.time()
    print("Build BTree time ", end_time - start_time)
    err = 0
    print("Calculate error")
    start_time = time.time()
    for ind in range(len(test_set_x)):
        pre = bt.predict(test_set_x[ind])
        err += abs(pre - test_set_y[ind])
    end_time = time.time()
    print("Search time ", (end_time - start_time) / len(test_set_x))
    print("mean error = ", err * 1.0 / len(test_set_x))
    print("*************end Learned NN************")


if __name__ == "__main__":
    train_index(Distribution.EXPONENTIAL)
