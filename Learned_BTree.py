# Main File for Learned Index

from __future__ import print_function
import pandas as pd
from Trained_NN import TrainedNN, AbstractNN, ParameterPool, set_data_type
from btree import BTree
from data.create_data import create_data, Distribution
import time, gc, json
import os, sys, getopt

# Setting 
BLOCK_SIZE = 100
TOTAL_NUMBER = 300000

# data files
filePath = {
    Distribution.RANDOM: "data/random.csv",
    Distribution.BINOMIAL: "data/binomial.csv",
    Distribution.POISSON: "data/poisson.csv",
    Distribution.EXPONENTIAL: "data/exponential.csv",
    Distribution.NORMAL: "data/normal.csv",
    Distribution.LOGNORMAL: "data/lognormal.csv"

}

# result record path
pathString = {
    Distribution.RANDOM: "Random",
    Distribution.BINOMIAL: "Binomial",
    Distribution.POISSON: "Poisson",
    Distribution.EXPONENTIAL: "Exponential",
    Distribution.NORMAL: "Normal",
    Distribution.LOGNORMAL: "Lognormal"
}

# threshold for train (judge whether stop train and replace with BTree)
thresholdPool = {
    Distribution.RANDOM: [1, 4],    
    Distribution.EXPONENTIAL: [55, 10000]
}   

# whether use threshold to stop train for models in stages
useThresholdPool = {
    Distribution.RANDOM: [True, False],    
    Distribution.EXPONENTIAL: [True, False],    
}

# hybrid training structure, 2 stages
def hybrid_training(threshold, use_threshold, stage_nums, core_nums, train_step_nums, batch_size_nums, learning_rate_nums,
                    keep_ratio_nums, train_data_x, train_data_y, test_data_x, test_data_y):
    stage_length = len(stage_nums)
    col_num = stage_nums[1]
    # initial
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
                # first stage, calculate how many models in next stage
                divisor = stage_nums[i + 1] * 1.0 / (TOTAL_NUMBER / BLOCK_SIZE)
                for k in tmp_labels[i][j]:
                    labels.append(int(k * divisor))
                for k in test_data_y:
                    test_labels.append(int(k * divisor))
            else:
                labels = tmp_labels[i][j]
                test_labels = test_data_y    
            # train model                    
            tmp_index = TrainedNN(threshold[i], use_threshold[i], core_nums[i], train_step_nums[i], batch_size_nums[i],
                                    learning_rate_nums[i],
                                    keep_ratio_nums[i], inputs, labels, test_inputs, test_labels)            
            tmp_index.train()      
            # get parameters in model (weight matrix and bias matrix)      
            index[i][j] = AbstractNN(tmp_index.get_weights(), tmp_index.get_bias(), core_nums[i], tmp_index.cal_err())
            del tmp_index
            gc.collect()
            if i < stage_length - 1:
                # allocate data into training set for models in next stage
                for ind in range(len(tmp_inputs[i][j])):
                    # pick model in next stage with output of this model
                    p = index[i][j].predict(tmp_inputs[i][j][ind])                    
                    if p > stage_nums[i + 1] - 1:
                        p = stage_nums[i + 1] - 1
                    tmp_inputs[i + 1][p].append(tmp_inputs[i][j][ind])
                    tmp_labels[i + 1][p].append(tmp_labels[i][j][ind])

    for i in range(stage_nums[stage_length - 1]):
        if index[stage_length - 1][i] is None:
            continue
        mean_abs_err = index[stage_length - 1][i].mean_err
        if mean_abs_err > threshold[stage_length - 1]:
            # replace model with BTree if mean error > threshold
            print("Using BTree")
            index[stage_length - 1][i] = BTree(2)
            index[stage_length - 1][i].build(tmp_inputs[stage_length - 1][i], tmp_labels[stage_length - 1][i])
    return index

# main function for training idnex
def train_index(threshold, use_threshold, distribution, path):
    # data = pd.read_csv("data/random_t.csv", header=None)
    # data = pd.read_csv("data/exponential_t.csv", header=None)
    data = pd.read_csv(path, header=None)
    train_set_x = []
    train_set_y = []
    test_set_x = []
    test_set_y = []

    set_data_type(distribution)
    # read parameter
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
    # set number of models for second stage (1 model deal with 10000 records)
    stage_set[1] = int(round(data.shape[0] / 10000))
    core_set = parameter.core_set
    train_step_set = parameter.train_step_set
    batch_size_set = parameter.batch_size_set
    learning_rate_set = parameter.learning_rate_set
    keep_ratio_set = parameter.keep_ratio_set

    global TOTAL_NUMBER
    TOTAL_NUMBER = data.shape[0]
    for i in range(data.shape[0]):
        train_set_x.append(data.ix[i, 0])
        train_set_y.append(data.ix[i, 1])
        #train_set_x.append(data.ix[i, 0])
        #train_set_y.append(data.ix[i, 1])

    test_set_x = train_set_x[:]
    test_set_y = train_set_y[:]     
    # data = pd.read_csv("data/random_t.csv", header=None)
    # data = pd.read_csv("data/exponential_t.csv", header=None)
    # for i in range(data.shape[0]):
    #     test_set_x.append(data.ix[i, 0])
    #     test_set_y.append(data.ix[i, 1])

    print("*************start Learned NN************")
    print("Start Train")
    start_time = time.time()
    # train index
    trained_index = hybrid_training(threshold, use_threshold, stage_set, core_set, train_step_set, batch_size_set, learning_rate_set,
                                    keep_ratio_set, train_set_x, train_set_y, [], [])
    end_time = time.time()
    learn_time = end_time - start_time
    print("Build Learned NN time ", learn_time)
    print("Calculate Error")
    err = 0
    start_time = time.time()
    # calculate error
    for ind in range(len(test_set_x)):
        # pick model in next stage
        pre1 = trained_index[0][0].predict(test_set_x[ind])
        if pre1 > stage_set[1] - 1:
            pre1 = stage_set[1] - 1
        # predict position
        pre2 = trained_index[1][pre1].predict(test_set_x[ind])
        err += abs(pre2 - test_set_y[ind])
    end_time = time.time()
    search_time = (end_time - start_time) / len(test_set_x)
    print("Search time %f " % search_time)
    mean_error = err * 1.0 / len(test_set_x)
    print("mean error = ", mean_error)
    print("*************end Learned NN************\n\n")
    # write parameter into files
    result_stage1 = {0: {"weights": trained_index[0][0].weights, "bias": trained_index[0][0].bias}}
    result_stage2 = {}
    for ind in range(len(trained_index[1])):
        if trained_index[1][ind] is None:
            continue
        if isinstance(trained_index[1][ind], BTree):
            tmp_result = []
            for ind, node in trained_index[1][ind].nodes.items():
                item = {}
                for ni in node.items:
                    if ni is None:
                        continue
                    item = {"key": ni.k, "value": ni.v}
                tmp = {"index": node.index, "isLeaf": node.isLeaf, "children": node.children, "items": item,
                       "numberOfkeys": node.numberOfKeys}
                tmp_result.append(tmp)
            result_stage2[ind] = tmp_result
        else:
            result_stage2[ind] = {"weights": trained_index[1][ind].weights,
                                  "bias": trained_index[1][ind].weights}
    result = [{"stage": 1, "parameters": result_stage1}, {"stage": 2, "parameters": result_stage2}]

    with open("model/" + pathString[distribution] + "/full_train/NN/" + str(TOTAL_NUMBER) + ".json", "wb") as jsonFile:
        json.dump(result, jsonFile)

    # wirte performance into files
    performance_NN = {"type": "NN", "build time": learn_time, "search time": search_time, "average error": mean_error,
                      "store size": os.path.getsize(
                          "model/" + pathString[distribution] + "/full_train/NN/" + str(TOTAL_NUMBER) + ".json")}
    with open("performance/" + pathString[distribution] + "/full_train/NN/" + str(TOTAL_NUMBER) + ".json",
              "wb") as jsonFile:
        json.dump(performance_NN, jsonFile)

    del trained_index
    gc.collect()
    
    # build BTree index
    print("*************start BTree************")
    bt = BTree(2)
    print("Start Build")
    start_time = time.time()
    bt.build(test_set_x, test_set_y)
    end_time = time.time()
    build_time = end_time - start_time
    print("Build BTree time ", build_time)
    err = 0
    print("Calculate error")
    start_time = time.time()
    for ind in range(len(test_set_x)):
        pre = bt.predict(test_set_x[ind])
        err += abs(pre - test_set_y[ind])
        if err != 0:
            flag = 1
            pos = pre
            off = 1
            while pos != test_set_y[ind]:
                pos += flag * off
                flag = -flag
                off += 1            
    end_time = time.time()
    search_time = (end_time - start_time) / len(test_set_x)
    print("Search time ", search_time)
    mean_error = err * 1.0 / len(test_set_x)
    print("mean error = ", mean_error)
    print("*************end BTree************")

    # write BTree into files
    result = []
    for ind, node in bt.nodes.items():
        item = {}
        for ni in node.items:
            if ni is None:
                continue
            item = {"key": ni.k, "value": ni.v}
        tmp = {"index": node.index, "isLeaf": node.isLeaf, "children": node.children, "items": item,
               "numberOfkeys": node.numberOfKeys}
        result.append(tmp)

    with open("model/" + pathString[distribution] + "/full_train/BTree/" + str(TOTAL_NUMBER) + ".json",
              "wb") as jsonFile:
        json.dump(result, jsonFile)

    # write performance into files
    performance_BTree = {"type": "BTree", "build time": build_time, "search time": search_time,
                         "average error": mean_error,
                         "store size": os.path.getsize(
                             "model/" + pathString[distribution] + "/full_train/BTree/" + str(TOTAL_NUMBER) + ".json")}
    with open("performance/" + pathString[distribution] + "/full_train/BTree/" + str(TOTAL_NUMBER) + ".json",
              "wb") as jsonFile:
        json.dump(performance_BTree, jsonFile)

    del bt
    gc.collect()


# Main function for sampel training
def sample_train(threshold, use_threshold, distribution, training_percent, path):
    data = pd.read_csv(path, header=None)
    train_set_x = []
    train_set_y = []
    test_set_x = []
    test_set_y = []

    set_data_type(distribution)
    #read parameters
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
    stage_set[1] = int(data.shape[0] * training_percent / 10000)
    core_set = parameter.core_set
    train_step_set = parameter.train_step_set
    batch_size_set = parameter.batch_size_set
    learning_rate_set = parameter.learning_rate_set
    keep_ratio_set = parameter.keep_ratio_set

    global TOTAL_NUMBER
    TOTAL_NUMBER = data.shape[0]
    interval = int(1 / training_percent)
    # pick data for training according to training percent
    if training_percent != 0.8:
        for i in range(TOTAL_NUMBER):
            test_set_x.append(data.ix[i, 0])
            test_set_y.append(data.ix[i, 1])
            if i % interval == 0:
                train_set_x.append(data.ix[i, 0])
                train_set_y.append(data.ix[i, 1])
    else:
        for i in range(TOTAL_NUMBER):
            test_set_x.append(data.ix[i, 0])
            test_set_y.append(data.ix[i, 1])
            if i % 5 != 0:
                train_set_x.append(data.ix[i, 0])
                train_set_y.append(data.ix[i, 1])

    print("*************start Learned NN************")
    print("Start Train")
    start_time = time.time()
    trained_index = hybrid_training(threshold, use_threshold, stage_set, core_set, train_step_set, batch_size_set, learning_rate_set,
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
    result_stage1 = {0: {"weights": trained_index[0][0].weights, "bias": trained_index[0][0].bias}}
    result_stage2 = {}
    for ind in range(len(trained_index[1])):
        if trained_index[1][ind] is None:
            continue
        if isinstance(trained_index[1][ind], BTree):
            tmp_result = []
            for ind, node in trained_index[1][ind].nodes.items():
                item = {}
                for ni in node.items:
                    if ni is None:
                        continue
                    item = {"key": ni.k, "value": ni.v}
                tmp = {"index": node.index, "isLeaf": node.isLeaf, "children": node.children, "items": item,
                       "numberOfkeys": node.numberOfKeys}
                tmp_result.append(tmp)
            result_stage2[ind] = tmp_result
        else:
            result_stage2[ind] = {"weights": trained_index[1][ind].weights,
                                  "bias": trained_index[1][ind].bias}
    result = [{"stage": 1, "parameters": result_stage1}, {"stage": 2, "parameters": result_stage2}]

    with open("model/" + pathString[distribution] + "/sample_train/NN/" + str(training_percent) + ".json",
              "wb") as jsonFile:
        json.dump(result, jsonFile)

    performance_NN = {"type": "NN", "build time": learn_time, "search time": search_time, "average error": mean_error,
                      "store size": os.path.getsize(
                          "model/" + pathString[distribution] + "/sample_train/NN/" + str(training_percent) + ".json")}
    with open("performance/" + pathString[distribution] + "/sample_train/NN/" + str(training_percent) + ".json",
              "wb") as jsonFile:
        json.dump(performance_NN, jsonFile)

    del trained_index
    gc.collect()

# help message
def show_help_message(msg):
    help_message = {'command': 'python Learned_BTree.py -t <Type> -d <Distribution> [-p|-n] [Percent]|[Number] [-c] [New data] [-h]',
                    'type': 'Type: sample, full',
                    'distribution': 'Distribution: random, exponential',
                    'percent': 'Percent: 0.1-1.0, default value = 0.5; sample train data size = 300,000',
                    'number': 'Number: 10,000-1,000,000, default value = 300,000',
                    'new data': 'New Data: INTEGER, 0 for no creating new data file, others for creating, default = 1',
                    'fpError': 'Percent cannot be assigned in full train.',
                    'snError': 'Number cannot be assigned in sample train.',
                    'noTypeError': 'Please choose the type first.',
                    'noDistributionError': 'Please choose the distribution first.'}
    help_message_key = ['command', 'type', 'distribution', 'percent', 'number', 'new data']
    if msg == 'all':
        for k in help_message_key:
            print(help_message[k])

    else:
        print(help_message['command'])
        print('Error! ' + help_message[msg])

# command line
def main(argv):
    distribution = None
    per = 0.5
    num = 300000
    is_sample = False
    is_type = False
    is_distribution = False
    do_create = True
    try:
        opts, args = getopt.getopt(argv, "hd:t:p:n:c:")
    except getopt.GetoptError:
        show_help_message('command')
        sys.exit(2)
    for opt, arg in opts:
        arg = str(arg).lower()
        if opt == '-h':
            show_help_message('all')
            return
        elif opt == '-t':
            if arg == "sample":
                is_sample = True
                is_type = True
            elif arg == "full":
                is_sample = False
                is_type = True
            else:
                show_help_message('type')
                return
        elif opt == '-d':
            if not is_type:
                show_help_message('noTypeError')
                return
            if arg == "random":
                distribution = Distribution.RANDOM
                is_distribution = True
            elif arg == "exponential":
                distribution = Distribution.EXPONENTIAL
                is_distribution = True
            else:
                show_help_message('distribution')
                return
        elif opt == '-p':
            if not is_type:
                show_help_message('noTypeError')
                return
            if not is_distribution:
                show_help_message('noDistributionError')
                return
            per = float(arg)
            if not 0.1 <= per <= 1.0:
                show_help_message('percent')
                return

        elif opt == '-n':
            if not is_type:
                show_help_message('noTypeError')
                return
            if not is_distribution:
                show_help_message('noDistributionError')
                return
            if is_sample:
                show_help_message('snError')
                return
            num = int(arg)
            if not 10000 <= num <= 1000000:
                show_help_message('number')
                return

        elif opt == '-c':
            if not is_distribution:
                show_help_message('noDistributionError')
                return
            do_create = not (int(arg) == 0)

        else:
            print("Unknown parameters, please use -h for instructions.")
            return

    if not is_type:
        show_help_message('noTypeError')
        return
    if not is_distribution:
        show_help_message('noDistributionError')
        return
    if do_create:
        create_data(distribution, num)
    if is_sample:        
        sample_train(thresholdPool[distribution], useThresholdPool[distribution], distribution, per, filePath[distribution])
    else:
        train_index(thresholdPool[distribution], useThresholdPool[distribution], distribution, filePath[distribution])


if __name__ == "__main__":
    main(sys.argv[1:])
