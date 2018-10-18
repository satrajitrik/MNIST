from __future__ import division
import scipy
import numpy as np
import load_dataset as ld
import matplotlib.pyplot as plt

train_data = ld.read()
alpha = 0.001
y_train_data = train_data[0]
x_train_data = train_data[1].reshape([60000, 28 * 28]) / 255

test_data = ld.read('testing')
y_test_data = test_data[0]
x_test_data = test_data[1].reshape([10000, 28 * 28]) / 255

weights = np.tile(np.zeros(784), (9, 1))

def post_prob(x, y):
    if y == 9:
        return 1 / sum_exps(x)
    else:
        return np.exp(np.dot(weights[y], x)) / sum_exps(x)

def sum_exps(x):
    exp_array = []
    for weight in weights:
        exp_array.append(np.exp(np.dot(weight, x)))
    return (1 + np.sum(exp_array))

def gradient_ascent(p, y, x):
    for i in range(len(weights)):
        if y == i:
            weights[i] = np.add(np.array(weights[i]), np.array(alpha * (1 - p[i]) * np.array(x)))
        else:
            weights[i] = np.add(np.array(weights[i]), np.array(alpha * (-1) * p[i] * np.array(x)))

def train():
    for i in range(len(x_train_data)):
        predictions = []
        for j in range(10):
            predictions.append(round(post_prob(x_train_data[i], j), 2))
        gradient_ascent(predictions, y_train_data[i], x_train_data[i])


def test():
    predictions = []
    for i in range(len(x_test_data)):
        p = []
        for j in range(10):
            p.append(round(post_prob(x_test_data[i], j), 2))
        index = np.argmax(p)
        predictions.append(index)
    count = 0
    for i in range(len(predictions)):
        if predictions[i] == y_test_data[i]:
            count += 1
    return (count / 10000)

def main():
    accuracy_list = []
    size_list = []
    for i in range(100):
        train()
        accuracy = test()
        print("For iteration = ", i + 1, " accuracy = ", accuracy * 100)
        accuracy_list.append(accuracy * 100)
        size_list.append((i + 1))

    plt.plot(size_list, accuracy_list, 'ro')
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.show()

if __name__ == "__main__":
    main()
