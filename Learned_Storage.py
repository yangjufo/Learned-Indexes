import pandas as pd
from Trained_NN import TrainedNN, ParameterPool, set_data_type
from btree import BTree
from data.create_data import create_data_hash, Distribution
import time, json, csv, math

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

    return trained_index


def optimize_storage(threshold, data_part_distance, learning_percent, distribution):
    store_path = storePath[distribution]
    to_store_path = toStorePath[distribution]

    tmp_data = pd.read_csv(store_path, header=None)
    train_set_x = []
    train_set_y = []
    test_set_x = []
    test_set_y = []
    global STORE_NUMBER
    STORE_NUMBER = tmp_data.shape[0]
    for i in range(STORE_NUMBER):
        # test_set_x.append(tmp_data.ix[i, 0])
        # test_set_y.append(tmp_data.ix[i, 1])
        train_set_x.append(tmp_data.ix[i, 0])
        train_set_y.append(tmp_data.ix[i, 1])
    store_data = train_set_x[:]

    to_store_data = pd.read_csv(to_store_path, header=None)
    print("************Start Optimization**************")
    trained_index = learn_density(threshold, distribution, train_set_x, train_set_y, test_set_x, test_set_y)
    stage_size = int(STORE_NUMBER / 10000)
    min_value = train_set_x[0]
    max_value = train_set_x[-1]
    data_density = []
    data_density_block = [0]
    data_part_num = int(math.ceil((max_value - min_value) * 1.0 / data_part_distance))
    last_pre = 0
    store_block_num = int(math.ceil(len(store_data) * 1.0 / BLOCK_SIZE))
    start_time = time.time()
    for i in range(1, data_part_num + 1):
        pre_data = min_value + i * data_part_distance
        if pre_data > max_value:
            pre_data = max_value
        pre1 = trained_index[0][0].predict(pre_data)
        if pre1 > stage_size - 1:
            pre1 = stage_size - 1
        pre2 = trained_index[1][pre1].predict(pre_data)
        if pre2 > store_block_num:
            pre2 = store_block_num
        if pre2 < last_pre:
            pre2 = last_pre
        data_density_block.append(pre2)
        data_density.append(abs(pre2 - last_pre) * 1.0 / store_block_num)
        last_pre = pre2

    move_steps = 0
    store_data = train_set_x[:]
    total_block_num = int(math.ceil((to_store_data.shape[0] + len(train_set_x)) * 1.0 / BLOCK_SIZE))
    for i in range(to_store_data.shape[0] + BLOCK_SIZE):
        store_data.append(-1)
    block_pos = total_block_num
    data_optimization_block = []
    for i in range(data_part_num, 0, -1):
        block_pos -= int(round(data_density[i - 1] * total_block_num))
        data_optimization_block.insert(0, block_pos)
        for j in range(data_density_block[i - 1], data_density_block[i] + 1):
            old_block_pos = j * BLOCK_SIZE
            if old_block_pos == 0:
                break
            new_block_pos = (block_pos + j - data_density_block[i - 1]) * BLOCK_SIZE
            if old_block_pos >= new_block_pos:
                continue
            k = 0
            for k in range(BLOCK_SIZE):
                if store_data[old_block_pos + k] == -1:
                    break
                store_data[new_block_pos + k] = store_data[old_block_pos + k]
                store_data[old_block_pos + k] = -1
            move_steps += k
    end_time = time.time()
    average_optimize_time = (end_time - start_time) * 1.0 / to_store_data.shape[0]
    print("Average Optimize Time: %f" % average_optimize_time)

    with open("tmp_result.csv", 'wb') as csvFile:
        csv_writer = csv.writer(csvFile)
        for d in store_data:
            csv_writer.writerow([d])

    print("************With Optimization**************")
    pos = 0
    ins_pos = 0
    start_time = time.time()
    count = 0
    for i in range(to_store_data.shape[0]):
        pre_data = to_store_data.ix[i, 0]
        if pre_data / data_part_distance == count:
            count += 1
            for j in range(data_part_num - 1):
                pos = data_optimization_block[j] * BLOCK_SIZE
                if store_data[pos] > pre_data:
                    if j != 0:
                        pos = data_optimization_block[j - 1] * BLOCK_SIZE
                    else:
                        pos = 0
                    break
        while (store_data[pos] > pre_data or store_data[pos] == -1) and pos > 0:
            pos -= 1
        while store_data[pos] <= pre_data and pos < len(store_data) - 1:
            pos += 1        
        if ins_pos < pos:
            ins_pos = pos
        while store_data[ins_pos] != -1 and ins_pos < len(store_data) - 1:
            ins_pos += 1
        if ins_pos == len(store_data) - 1:
            store_data.append(-1)
            ins_pos += 1
        for j in range(ins_pos, pos, -1):
            store_data[j] = store_data[j - 1]
        store_data[pos] = pre_data
        move_steps += ins_pos - pos
    end_time = time.time()
    average_move_steps = (move_steps * 1.0 / to_store_data.shape[0])
    average_move_time = (end_time - start_time) * 1.0 / to_store_data.shape[0]
    print("Average Move Steps: %f" % average_move_steps)
    print("Average Move Time: %f" % average_move_time)
    result = [{"Average Moving Steps": average_move_steps, "Average Moving Time": average_move_time,
               "Average Optimizing Time": average_optimize_time}]
    with open("store_performance/" + pathString[distribution] + "/optimization/" + str(data_part_distance) + "_" + str(
            learning_percent) + ".json", "wb") as jsonFile:
        json.dump(result, jsonFile)

    with open("test_result.csv", 'wb') as csvFile:
        csv_writer = csv.writer(csvFile)
        for d in store_data:
            csv_writer.writerow([d])

    # print("************Without Optimization**************")
    # store_data = train_set_x[:]
    # move_steps = 0
    # start_time = time.time()
    # pos = 0
    # for i in range(to_store_data.shape[0]):
    #     pre_data = to_store_data.ix[i, 0]
    #     while store_data[pos] > pre_data and pos > 0:
    #         pos -= 1
    #     while store_data[pos] < pre_data and pos < len(store_data) - 1:
    #         pos += 1
    #     store_data.append(-1)
    #     for j in range(len(store_data) - 1, pos, -1):
    #         store_data[j] = store_data[j - 1]
    #     store_data[pos] = pre_data
    #     move_steps += len(store_data) - 1 - pos
    # end_time = time.time()
    # average_move_steps = (move_steps * 1.0 / to_store_data.shape[0])
    # average_move_time = (end_time - start_time) * 1.0 / to_store_data.shape[0]
    # print("Average Move Steps: %f" % average_move_steps)
    # print("Average Move Time: %f" % average_move_time)

    # result = [{"Average Moving Steps": average_move_steps, "Average Moving Time": average_move_time}]

    # with open("store_performance/" + pathString[distribution] + "/no_optimization/" 
    # + str(learning_percent) + ".json", "wb") as jsonFile:
    #     json.dump(result, jsonFile)


if __name__ == "__main__":
    learning_percent = [0.3, 0.5, 0.8]
    learning_ind = 2
    # create_data_hash(Distribution.RANDOM, learning_percent[learning_ind])
    data_distance = [100000, 1000000, 10000000, 100000000]
    data_distance_random = [100, 1000, 10000, 100000]
    distance_ind = 0
    optimize_storage(1, data_distance_random[distance_ind], learning_percent[learning_ind], Distribution.RANDOM)
