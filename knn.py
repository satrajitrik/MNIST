from __future__ import division
import scipy
import numpy as np
import load_dataset as ld
import matplotlib.pyplot as plt
from scipy.spatial import distance


train_data = ld.read()
y_train_data = train_data[0]
x_train_data = train_data[1].reshape([60000, 28 * 28])

test_data = ld.read('testing')
y_test_data = test_data[0]
x_test_data = test_data[1].reshape([10000, 28 * 28])

distance_list = distance.cdist(x_test_data,x_train_data)

def most_common(list):
    return max(set(list), key = list.count)

def knn(k):
    count = 0
    for i in range(len(distance_list)):
        if most_common(list(y_train_data[np.argpartition(distance_list[i],k)[:k]])) == y_test_data[i]:
            count += 1

    accuracy = count / 10000
    print("For k = ", str(k), " accuracy ", accuracy)
    return accuracy

def main():
    k_list = [1, 3, 5, 10, 30, 50, 70, 80, 90, 100]
    accuracy_list=[]
    for k in k_list:
        accuracy_list.append(knn(k))

    plt.plot(k_list, list(np.array(accuracy_list)*100), "ro")
    plt.xlabel("K")
    plt.ylabel("Accuracy")

if __name__ == "__main__":
    main()
