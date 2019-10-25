import pickle
import gzip
import numpy as np


def activate(vec):
    best_score = vec[0]
    idx = 0
    for i in range(1, len(vec)):
        if vec[i] > best_score:
            best_score = vec[i]
            idx = i
    ar = np.zeros((10, 1))
    ar[idx] = 1
    return ar


def reshaped_result(result):
    ar = np.zeros((10, 1))
    ar[result] = 1
    return ar


def process(data, proc_weights, proc_bias, result, rate=None):
    proc_data = np.asmatrix(data)
    proc_result = reshaped_result(result)
    z = np.dot(proc_weights, np.transpose(proc_data)) + proc_bias
    output = activate(z)
    if rate is not None:
        proc_weights += np.dot(proc_result - output, proc_data) * rate
        proc_bias += (proc_result - output) * rate
    return np.array_equal(output, proc_result)


def train(train_weights, train_bias, learning_rate):
    global train_set

    nr_iterations = 1000

    # the number of records
    data_size = len(train_set[0])

    all_classified = False
    while not all_classified and nr_iterations > 0:
        all_classified = True
        for i in range(0, data_size):
            if not process(train_set[0][i], train_weights, train_bias, train_set[1][i], rate=learning_rate):
                all_classified = False
        nr_iterations -= 1
        print(nr_iterations)


def check_set(checked_weights, checked_bias, data_set):
    hit = 0
    data_size = len(data_set[0])
    for i in range(0, data_size):
        if process(data_set[0][i], checked_weights, checked_bias, data_set[1][i]):
            hit += 1

    return hit / data_size


def validate(val_weighs, val_bias):
    global valid_set
    return check_set(val_weighs, val_bias, valid_set)


def test(test_weights, test_bias):
    global test_set
    return check_set(test_weights, test_bias, test_set)


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
f.close()

learning_rate_pool = [0.1]
maxim = -1
best_weights = []
best_bias = []
best_lr = -1

# the number of attributes / dimensions
n = len(train_set[0][0])
weights = np.zeros((10, n))
bias = np.zeros((10, 1))

for lr in learning_rate_pool:

    weights.fill(0)
    bias.fill(0)

    train(weights, bias, lr)
    success = validate(weights, bias)
    if success > maxim:
        maxim = success
        best_weights = weights
        best_bias = bias
        best_lr = lr
    print("Validation score with learning rate = " + str(lr) + ": " + str(success))

print("Best validation score: " + str(maxim) + " (learning rate = " + str(best_lr) + ")")
print("Testing score: " + str(test(best_weights, best_bias)))

# Validation score with learning rate = 0.1: 0.8974
# Best validation score: 0.8974 (learning rate = 0.1)
# Testing score: 0.8933
