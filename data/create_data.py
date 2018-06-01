# Main file for create test data
from enum import Enum
import numpy as np
import csv
import random

SIZE = 100000
BLOCK_SIZE = 100


class Distribution(Enum):
    RANDOM = 0
    BINOMIAL = 1
    POISSON = 2
    EXPONENTIAL = 3
    NORMAL = 4
    LOGNORMAL = 5

# store path
filePath = {
    Distribution.RANDOM: "data/random.csv",
    Distribution.BINOMIAL: "data/binomial.csv",
    Distribution.POISSON: "data/poisson.csv",
    Distribution.EXPONENTIAL: "data/exponential.csv",
    Distribution.NORMAL: "data/normal.csv",
    Distribution.LOGNORMAL: "data/lognormal.csv"
}

# path for storage optimization
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

# create data
def create_data(distribution, data_size=SIZE):
    if distribution == Distribution.RANDOM:
        data = random.sample(range(data_size * 2), data_size)
    elif distribution == Distribution.BINOMIAL:
        data = np.random.binomial(100, 0.5, size=data_size)
    elif distribution == Distribution.POISSON:
        data = np.random.poisson(6, size=data_size)
    elif distribution == Distribution.EXPONENTIAL:
        data = np.random.exponential(10, size=data_size)
    elif distribution == Distribution.LOGNORMAL:
        data = np.random.lognormal(0, 2, data_size)
    else:
        data = np.random.normal(1000, 100, size=data_size)
    res_path = filePath[distribution]
    data.sort()
    with open(res_path, 'wb') as csvFile:
        csv_writer = csv.writer(csvFile)
        i = 0
        if distribution == Distribution.EXPONENTIAL:
            for d in data:
                csv_writer.writerow([int(d * 10000000), i / BLOCK_SIZE])
                i += 1
        elif distribution == Distribution.LOGNORMAL:
            for d in data:
                csv_writer.writerow([int(d * 10000), i / BLOCK_SIZE])
                i += 1
        else:
            for d in data:
                csv_writer.writerow([int(d), i / BLOCK_SIZE])
                i += 1


def create_data_storage(distribution, learning_percent=0.5, data_size=SIZE):
    if distribution == Distribution.RANDOM:
        data = random.sample(range(data_size * 2), data_size)
    elif distribution == Distribution.BINOMIAL:
        data = np.random.binomial(100, 0.5, size=data_size)
    elif distribution == Distribution.POISSON:
        data = np.random.poisson(6, size=data_size)
    elif distribution == Distribution.EXPONENTIAL:
        data = np.random.exponential(10, size=data_size)
    elif distribution == Distribution.LOGNORMAL:
        data = np.random.lognormal(0, 2, data_size)
    else:
        data = np.random.normal(1000, 100, size=data_size)
    store_path = storePath[distribution]
    to_store_path = toStorePath[distribution]
    data.sort()
    store_bits = []
    if learning_percent == 0.8:
        for i in range(data_size):
            store_bits.append(random.randint(0, 4))
    else:
        for i in range(data_size):
            store_bits.append(random.randint(0, int(1.0 / learning_percent) - 1))
    i = 0
    insert_data = []
    with open(store_path, 'wb') as csvFile:
        csv_writer = csv.writer(csvFile)
        #deal with sample training and storage optimization
        if distribution == Distribution.EXPONENTIAL:
            if learning_percent == 0.8:
                for ind in range(data_size):
                    din = int(data[ind] * 10000000)
                    if store_bits[ind] != 0:                        
                        csv_writer.writerow([din, i / BLOCK_SIZE])
                        i += 1
                    else:
                        insert_data.append(din)
            else:
                for ind in range(data_size):
                    din = int(data[ind] * 10000000)
                    if store_bits[ind] == 0:                        
                        csv_writer.writerow([din, i / BLOCK_SIZE])
                        i += 1
                    else:
                        insert_data.append(din)
        elif distribution == Distribution.LOGNORMAL:
            for d in data:
                din = int(d * 10000)
        else:
            if learning_percent == 0.8:
                for ind in range(data_size):
                    din = int(data[ind])
                    if store_bits[ind] != 0:                        
                        csv_writer.writerow([din, i / BLOCK_SIZE])
                        i += 1
                    else:
                        insert_data.append(din)
            else:
                for ind in range(data_size):
                    din = int(data[ind])
                    if store_bits[ind] == 0:                        
                        csv_writer.writerow([din, i / BLOCK_SIZE])
                        i += 1
                    else:
                        insert_data.append(din)
    random.shuffle(insert_data) 
    with open(to_store_path, 'wb') as csvFile:
        csv_writer = csv.writer(csvFile)
        for din in insert_data:
            csv_writer.writerow([din])


if __name__ == "__main__":
    create_data_storage(Distribution.EXPONENTIAL)
