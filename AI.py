import numpy as np

HIDDEN_LAYER_RATIO = 0.3
RANDOM_SEED = 101
ITERATIONS = 100000
LEARNING_RATE = 0.01
DEBUG = True

weights1 = []
weights2 = []


def randomize(weights, number_of_neurons, number_of_inputs_per_neuron):
    w = np.random.random((number_of_inputs_per_neuron, number_of_neurons))
    for i in w:
        weights.append(i)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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
    layer2_delta = layer2_error * LEARNING_RATE

    layer1_error = layer2_delta.dot(weights2.T)
    layer1_delta = layer1_error * LEARNING_RATE

    layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
    layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

    weights1 += layer1_adjustment
    weights2 += layer2_adjustment


def train(training_set_inputs, training_set_outputs, number_of_training_iterations):
    global weights1
    global weights2

    for _ in range(number_of_training_iterations):
        output_from_layer_1, output_from_layer_2 = forward(training_set_inputs)
        backward(training_set_inputs, training_set_outputs, output_from_layer_1, output_from_layer_2)


def answer(probe, output_size):
    hidden_state, output = forward(probe)
    ans = 0
    maxim = -1e9
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
    train(np.array(inputs_list), np.array(outputs_list), ITERATIONS)
    _print(weights1)
    _print(weights2)

    _print("3) Compute average performance")
    hit = 0
    for i in range(len(inputs_list)):
        solution = answer(np.array(inputs_list[i]), output_size)
        if solution == outputs_list[i]:
            hit += 1
    _print(hit / len(inputs_list))
    
    return hit / len(inputs_list)


def cross_validation(input_size, output_size, entries, inputs_list, outputs_list):
    global nr_inputs
    nr_inputs = entries - 1
    hit = 0
    for i in range(0, entries):
        local_inputs = inputs_list[:i] + inputs_list[i+1:]
        local_outputs = outputs_list[:i] + outputs_list[i+1:]
        ratio = run(input_size, output_size, nr_inputs, local_inputs, local_outputs)
        print("Ratio for run " + str(i) + ": " + str(ratio))
        solution = answer(np.array(inputs_list[i]), output_size)
        if solution == outputs_list[i]:
            hit += 1
    return hit // nr_inputs


with open("segments.data", "r") as input_file:
    input_text = input_file.read().splitlines()

    meta = list(map(int, input_text[0].split()))

    input_size = meta[0]
    output_size = meta[1]
    hidden_layer_size = int(input_size * HIDDEN_LAYER_RATIO)
    nr_inputs = meta[2]

    inputs_list = [list(map(int, (i.split(', ')[:input_size]))) for i in input_text[1:]]
    outputs_list = [list(map(int, (i.split(', ')[input_size:]))) for i in input_text[1:]]


np.random.seed(RANDOM_SEED)

print("Starting cross-validation:")
performance = cross_validation(input_size, output_size, nr_inputs, inputs_list, outputs_list)
print("Ratio for the cross-validation: " + str(performance))

print("Starting full-validation:")
performance = run(input_size, output_size, nr_inputs, inputs_list, outputs_list)
print("Ratio for full-validation: " + str(performance))


