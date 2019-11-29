import pickle
import gzip
import numpy as np

HIDDEN_LAYER_RATIO = 0.2
RANDOM_SEED = 123
ITERATIONS = 1000
LEARNING_RATE = 0.01
STEP_ITERATION = 10
DEBUG = True

weights1 = []
weights2 = []
input_size = 784
output_size = 10


def randomize(weights, number_of_neurons, number_of_inputs_per_neuron):
    w = np.random.random((number_of_inputs_per_neuron, number_of_neurons))
    for i in w:
        weights.append(i)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    if LEARNING_RATE != 0:
        return LEARNING_RATE
    return x * (1-x)


def forward(inputs):
    global weights1
    global weights2

    output_from_layer1 = sigmoid(np.dot(inputs, weights1))
    output_from_layer2 = sigmoid(np.dot(output_from_layer1, weights2))
    return output_from_layer1, output_from_layer2


def backward(training_set_inputs, training_set_outputs, output_from_layer_1, output_from_layer_2):
    global weights1
    global weights2

    layer2_error = training_set_outputs - output_from_layer_2
    layer2_delta = layer2_error * sigmoid_derivative(output_from_layer_2)

    layer1_error = layer2_delta.dot(weights2.T)
    layer1_delta = layer1_error * sigmoid_derivative(output_from_layer_1)

    layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
    layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

    weights1 += layer1_adjustment
    weights2 += layer2_adjustment


def train(training_set_inputs, training_set_outputs, number_of_training_iterations):
    global weights1
    global weights2

    for i in range(number_of_training_iterations):
        output_from_layer_1, output_from_layer_2 = forward(training_set_inputs)
        backward(training_set_inputs, training_set_outputs, output_from_layer_1, output_from_layer_2)
        if i % STEP_ITERATION == 0:
            print("Iteration: " + str(i))


def answer(probe, output_size):
    hidden_state, output = forward(probe)
    ans = 0
    maxim = output[0]
    for i in range(len(output)):
        if output[i] > maxim:
            maxim = output[i]
            ans = i

    sol = [0 for _ in range(0, output_size)]
    sol[ans] = 1
    return sol


def _print(content):
    if DEBUG:
        print(content)


def run(input_size, output_size, nr_inputs, inputs_list, outputs_list):
    global weights1
    global weights2

    _print("1) Randomize")
    weights1 = []
    weights2 = []
    randomize(weights1, hidden_layer_size, input_size)
    randomize(weights2, output_size, hidden_layer_size)
    weights1 = np.array(weights1)
    weights2 = np.array(weights2)
    _print(weights1)
    _print(weights2)

    _print("2) Train")
    train(inputs_list, outputs_list, ITERATIONS)
    _print(weights1)
    _print(weights2)

    _print("3) Compute average performance")
    hit = 0
    for i in range(len(inputs_list)):
        solution = answer(inputs_list[i], output_size)
        if np.array_equal(np.array(solution), outputs_list[i]):
            hit += 1
    _print(hit / len(inputs_list))

    return hit / len(inputs_list)


def cross_validation(input_size, output_size, entries, inputs_list, outputs_list, only):
    global nr_inputs
    nr_inputs = entries - 1
    hit = 0
    for i in range(0, only):
        local_inputs = inputs_list[:i] + inputs_list[i+1:]
        local_outputs = outputs_list[:i] + outputs_list[i+1:]
        ratio = run(input_size, output_size, nr_inputs, local_inputs, local_outputs)
        print("Ratio for run " + str(i) + ": " + str(ratio))
        solution = answer(np.array(inputs_list[i]), output_size)
        if np.array_equal(np.array(solution), outputs_list[i]):
            hit += 1
    return hit / nr_inputs


def digit_to_list(x):
    global output_size

    l = [0 for _ in range(output_size)]
    l[x] = 1
    return l


hidden_layer_size = int(input_size * HIDDEN_LAYER_RATIO)
hidden_layer_size = 2


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
f.close()

inputs_list = np.concatenate((train_set[0], valid_set[0]))
nr_inputs = len(inputs_list)

outs = []
for i in np.concatenate((train_set[1], valid_set[1])):
    lst = digit_to_list(i)
    outs.append(lst)
outputs_list = np.array(outs)

print(inputs_list)
print(outputs_list)

np.random.seed(RANDOM_SEED)

#print("Starting cross-validation:")
#performance = cross_validation(input_size, output_size, nr_inputs, inputs_list, outputs_list, 5)
#print("Ratio for the cross-validation: " + str(performance))

print("Starting full-validation:")
performance = run(input_size, output_size, nr_inputs, inputs_list, outputs_list)
print("Ratio for full-validation: " + str(performance))


