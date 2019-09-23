import os
import csv
import random

def hypothesis(theta: list, X: list) -> float:
    '''
    Returns the hypothesis function result for the given theta and X vectors.
    '''
    m = len(theta)
    ans = 0.0
    # print(theta)
    # print(X)

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

def apply(X: list, y: list) -> list:
    m = len(X)
    theta = [0] * (m + 1)
    hypo = [0] * m
    alpha = 0.05
    min_cost = float('inf')

    # Add bias
    for x in X:
        x.insert(0, 1)
    for i in range(m):
        hypo[i] = hypothesis(theta, X[i])

    while True:
        cur_cost = cost(y, theta, hypo)
        if cur_cost < min_cost:
            min_cost = cur_cost
            min_theta = theta
        else:
            break

        # Change theta using gradient descent
        for j in range(m + 1):
            gradient = 0
            for i in range(m):
                gradient += ((hypo[i] - y[i]) * X[j][i])
            theta[j] = theta[j] - ((alpha/m) * gradient)

    return min_theta

def predict(X: list, y: list, theta: list):
    m = len(X)
    hypo = [0] * m

    # Add bias
    for x in X:
        x.insert(0, 1)
    for i in range(m):
        hypo[i] = hypothesis(theta, X[i])

    path = os.path.join(os.path.join(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets'), 'redwine'), 'winequality-red.csv')
    with open(path, mode='w') as f:
        for i in range(m):
            f.write('Predicted: ' + str(hypo[i]) + ', Actual: ' + y[i] + ', Error: ' + str(y[i] - hypo[i]))

def run(path: str, split=0.7):
    X = []
    y = []
    data = []
    skip = True

    with open(path, newline='') as f:
        reader = csv.reader(f)

        #with open(os.path.join(os.path.dirname(path), 'debug.txt'), mode='w') as d:
        for row in reader:
            if skip:
                skip = False
                continue 
            data.append(row)
                # d.write(str(row))

    train_x, train_y, test_x, test_y = split_data(data, split)

    theta = apply(train_x, train_y)
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

if __name__ == '__main__':
    dirname = os.path.dirname(os.path.dirname(__file__))
    datasets = os.path.join(dirname, 'datasets')
    redwine = os.path.join(os.path.join(datasets, 'redwine'), 'winequality-red.csv')
    
    # print(os.getcwd())
    run(redwine)