import os
import csv
import random
from .helper import *

def print_array(arr: list):
    print('Length: ' + str(len(arr)))
    print(arr)
    print()

def hypothesis(X: list, theta: list) -> float:
    '''
    Returns the hypothesis function result for the given theta and X vectors.
    '''
    m = len(theta)
    ans = 0.0

    for i in range(m):
        ans += theta[i] * X[i]

    return ans

def cost(y: list, theta: list, hypo: list) -> float:
    m = len(y)
    ans = 0.0

    for i in range(m):
        ans += ((hypo[i] - y[i]) ** 2)
    
    ans /= (2 * m)
    return ans

def apply(X: list, y: list, num_iter=100) -> list:
    m = len(X[0])
    theta = [0] * (m + 1)
    hypo = [0] * len(y)
    min_cost = float('inf')

    for x in X:
        x.insert(0, 1)
    for i in range(m):
        hypo[i] = hypothesis(X[i], theta)

    hypo_ref = hypo[:]
    iter = 0

    alpha = 10 ** -8
    while alpha < 100:
        while iter < num_iter:
            cur_cost = cost(y, theta, hypo)

            min_cost = min(min_cost, cur_cost)
            if min_cost == cur_cost:
                min_theta = theta

            # Change theta using gradient descent
            for j in range(m + 1):
                gradient = 0
                for i in range(len(y)):
                    gradient += ((hypo[i] - y[i]) * X[i][j])
                theta[j] = theta[j] - ((alpha/m) * gradient)

            iter += 1

        hypo = hypo_ref[:]
        alpha += 10 ** -7

    return min_theta

def predict(X: list, y: list, theta: list):
    m = len(X)
    hypo = [0] * m

    # Add bias
    for x in X:
        x.insert(0, 1)
    for i in range(m):
        hypo[i] = hypothesis(X[i], theta)

    path = os.path.join(os.path.join(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets'), 'redwine'), 'raw-results.csv')
    with open(path, mode='w') as f:
        f.write('Predicted,Actual,Error')
        f.write('\n')
        for i in range(m):
            f.write(str(hypo[i]) + ',' + str(y[i]) + ',' + str(y[i] - hypo[i]))
            f.write('\n')

def run(path: str, split=0.7):
    X = []
    y = []
    data = []
    skip = True

    with open(path, newline='') as f:
        reader = csv.reader(f)

        for row in reader:
            if skip:
                skip = False
                continue 
            row = [float(x) for x in row]
            data.append(row)

    train_x, train_y, test_x, test_y = split_data(data, split)

    theta = apply(train_x, train_y, 400)
    proj_debug = get_debug_path()
    with open(proj_debug, mode='a') as f:
        f.write('theta\n')
        f.write('length: ' + str(len(theta)) + '\n')
        f.write(str(theta))

    predict(test_x, test_y, theta)


def split_data(data: list, split=0.7) -> tuple:
    random.shuffle(data)
    cut = int(len(data) * split)

    train = data[:cut]
    test = data[cut:]

    train_y = [t.pop() for t in train]
    train_x = train

    test_y = [t.pop() for t in test]
    test_x = test

    return (train_x, train_y, test_x, test_y)
    
def go():
    run(get_dataset_path())