from enum import Enum
import numpy as np
import csv
import random

SIZE = 300001
BLOCK_SIZE = 100


class Distribution(Enum):
    RANDOM = 0
    BINOMIAL = 1
    POISSON = 2
    EXPONENTIAL = 3
    NORMAL = 4
    LOGNORMAL = 5


filePath = {
    Distribution.RANDOM: "data/random.csv",
    Distribution.BINOMIAL: "data/binomial.csv",
    Distribution.POISSON: "data/poisson.csv",
    Distribution.EXPONENTIAL: "data/exponential.csv",
    Distribution.NORMAL: "data/normal.csv",
    Distribution.LOGNORMAL: "data/lognormal.csv"
}


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


if __name__ == "__main__":
    create_data(Distribution.RANDOM)
