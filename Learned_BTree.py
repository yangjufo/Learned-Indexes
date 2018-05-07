from __future__ import print_function
import pandas as pd
from Trained_NN import TrainedNN, ParameterPool, set_data_type
from btree import BTree
from data.create_data import create_data, Distribution
import time
import gc
import json
import os

BLOCK_SIZE = 100
TOTAL_NUMBER = 300000

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
    return index


def sample_train(distribution, training_percent, path):
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
    stage_set[1] = data.shape[0] / 10000
    core_set = parameter.core_set
    train_step_set = parameter.train_step_set
    batch_size_set = parameter.batch_size_set
    learning_rate_set = parameter.learning_rate_set
    keep_ratio_set = parameter.keep_ratio_set

    global TOTAL_NUMBER
    TOTAL_NUMBER = data.shape[0]
	interval = 1 / training_percent
    for i in range(TOTAL_NUMBER - 1):
        test_set_x.append(data.ix[i, 0])
        test_set_y.append(data.ix[i, 1])
		if i % interval == 0:
	        train_set_x.append(data.ix[i, 0])
    	    train_set_y.append(data.ix[i, 1])

    print("*************strat Learned NN************")
    print("Start Train")
    start_time = time.time()
    trained_index = hybrid_training(stage_set, core_set, train_step_set, batch_size_set, learning_rate_set,
                                    keep_ratio_set, train_set_x, train_set_y, test_set_x, test_set_y)
    end_time = time.time()
    learn_time = end_time - start_time
    print("Build Learned NN time ", learn_time)
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
    search_time = (end_time - start_time) / len(test_set_x)
    print("Search time ", search_time)
    mean_error = err * 1.0 / len(test_set_x)
    print("mean error = ", mean_error)
    print("*************end Learned NN************\n\n")
    result_stage1 = {0: {"weights": trained_index[0][0].get_weight(), "bias": trained_index[0][0].get_bias()}}
    result_stage2 = {}
    for ind in range(len(trained_index[1])):
        if trained_index[1][ind] is None:
            continue
        result_stage2[ind] = {"weights": trained_index[1][ind].get_weight(), "bias": trained_index[1][ind].get_bias()}
    result = [{"stage": 1, "parameters": result_stage1}, {"stage": 2, "parameters": result_stage2}]

    with open("model/NN/" + str(training_percent) + ".json", "wb") as jsonFile:
        json.dump(result, jsonFile)

    performance_NN = {"type": "NN", "build time": learn_time, "search time": search_time, "average error": mean_error,
                      "store size": os.path.getsize("model/NN/" + str(TOTAL_NUMBER) + ".json")}
    with open("sample_performance/NN/" + str(training_percent) + ".json", "wb") as jsonFile:
        json.dump(performance_NN, jsonFile)

    del trained_index
    gc.collect()




if __name__ == "__main__":
    percents = [0.1, 0.3, 0.5, 0.8, 1.0]
    for per in percents:
        train_index(Distribution.RANDOM, per, filePath[Distribution.RANDOM])
