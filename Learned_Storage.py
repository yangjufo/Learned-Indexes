import pandas as pd
from Trained_NN import TrainedNN, ParameterPool, set_data_type
from btree import BTree
from data.create_data import create_data_hash, Distribution
import time, os, sys, gc, json

STORE_NUMBER = 100001
BLOCK_SIZE = 100

storePath = {
    Distribution.RANDOM: "data/random_s.csv",
    Distribution.BINOMIAL: "data/binomial_s.csv",
    Distribution.POISSON: "data/poisson_s.csv",
    Distribution.EXPONENTIAL: "data/exponential_s.csv",
    Distribution.NORMAL: "data/normal_s.csv",
    Distribution.LOGNORMAL: "data/lognormal_s.csv"
}

toStorePath = {
    Distribution.RANDOM: "data/random_t.csv",
    Distribution.BINOMIAL: "data/binomial_t.csv",
    Distribution.POISSON: "data/poisson_t.csv",
    Distribution.EXPONENTIAL: "data/exponential_t.csv",
    Distribution.NORMAL: "data/normal_t.csv",
    Distribution.LOGNORMAL: "data/lognormal_t.csv"
}

pathString = {
    Distribution.RANDOM: "Random",
    Distribution.BINOMIAL: "Binomial",
    Distribution.POISSON: "Poisson",
    Distribution.EXPONENTIAL: "Exponential",
    Distribution.NORMAL: "Normal",
    Distribution.LOGNORMAL: "Lognormal"
}


def hybrid_training(threshold, stage_nums, core_nums, train_step_nums, batch_size_nums, learning_rate_nums,
                    keep_ratio_nums, train_data_x, train_data_y, test_data_x, test_data_y):
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
                divisor = stage_nums[i + 1] * 1.0 / (STORE_NUMBER / BLOCK_SIZE)
                for k in tmp_labels[i][j]:
                    labels.append(int(k * divisor))
                for k in test_data_y:
                    test_labels.append(int(k * divisor))
            else:
                labels = tmp_labels[i][j]
                test_labels = test_data_y
            if i == 0:
                index[i][j] = TrainedNN(1, core_nums[i], train_step_nums[i], batch_size_nums[i], learning_rate_nums[i],
                                        keep_ratio_nums[i], inputs, labels, test_inputs, test_labels)
            else:
                index[i][j] = TrainedNN(threshold, core_nums[i], train_step_nums[i], batch_size_nums[i],
                                        learning_rate_nums[i],
                                        keep_ratio_nums[i], inputs, labels, test_inputs, test_labels)
            index[i][j].train()

            if i < stage_length - 1:
                for ind in range(len(tmp_inputs[i][j])):
                    p = index[i][j].predict(tmp_inputs[i][j][ind])
                    if p > stage_nums[i + 1] - 1:
                        p = stage_nums[i + 1] - 1
                    tmp_inputs[i + 1][p].append(tmp_inputs[i][j][ind])
                    tmp_labels[i + 1][p].append(tmp_labels[i][j][ind])

    for i in range(stage_nums[stage_length - 1]):
        if index[stage_length - 1][i] is None:
            continue
        mean_abs_err = index[stage_length - 1][i].cal_err()
        if mean_abs_err > threshold:
            print("Using BTree")
            index[stage_length - 1][i] = BTree(2)
            index[stage_length - 1][i].build(tmp_inputs[stage_length - 1][i], tmp_labels[stage_length - 1][i])
    return index


def learn_density(threshold, distribution, train_set_x, train_set_y, test_set_x, test_set_y):
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
    stage_set[1] = int(STORE_NUMBER / 10000)
    core_set = parameter.core_set
    train_step_set = parameter.train_step_set
    batch_size_set = parameter.batch_size_set
    learning_rate_set = parameter.learning_rate_set
    keep_ratio_set = parameter.keep_ratio_set

    print("*************start Learned NN************")
    print("Start Train")
    start_time = time.time()
    trained_index = hybrid_training(threshold, stage_set, core_set, train_step_set, batch_size_set, learning_rate_set,
                                    keep_ratio_set, train_set_x, train_set_y, test_set_x, test_set_y)
    end_time = time.time()
    learn_time = end_time - start_time
    print("Build Learned NN time %f " % learn_time)
    print("*************end Learned NN************")

    min_value = train_set_x[0]
    max_value = train_set_y[-1]
    data_density = []
    data_part_distance = 10000
    data_part_num = (max_value - min_value) / data_part_distance
    for i in range(data_part_num):
        pre_data = min_value + i * data_part_distance
        pre1 = trained_index[0][0].predict(pre_data)
        if pre1 > stage_set[1] - 1:
            pre1 = stage_set[1] - 1
        pre2 = trained_index[1][pre1].predict(pre_data)
        data_density.append(pre2)
    return data_density


def optimize_storage(threshold, distribution):
    store_path = storePath[distribution]
    to_store_path = toStorePath[distribution]

    tmp_data = pd.read_csv(store_path)
    train_set_x = []
    train_set_y = []
    test_set_x = []
    test_set_y = []
    global STORE_NUMBER
    STORE_NUMBER = tmp_data.shape[0]
    store_data = tmp_inputs = [[] for i in range(STORE_NUMBER / BLOCK_SIZE + 1)]
    for i in range(STORE_NUMBER):
        # test_set_x.append(tmp_data.ix[i, 0])
        # test_set_y.append(tmp_data.ix[i, 1])
        train_set_x.append(tmp_data.ix[i, 0])
        train_set_y.append(tmp_data.ix[i, 1])
        store_data[tmp_data.ix[i, 1]].append(tmp_data.ix[i,0])

    data_density = learn_density(threshold, distribution, train_set_x, train_set_y, test_set_x, test_set_y)
    to_store_data = pd.read_csv(to_store_path)
    to_store_number = to_store_data.shape[0]
    print("************Without Optimization**************")
    print("************With Optimization**************")
    



    return


if __name__ == "__main__":
    optimize_storage(1, Distribution.RANDOM)
