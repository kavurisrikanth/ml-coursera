import os
import csv
import random

def print_array(arr: list):
    print('Length: ' + str(len(arr)))
    print(arr)
    print()

def hypothesis(theta: list, X: list) -> float:
    '''
    Returns the hypothesis function result for the given theta and X vectors.
    '''
    m = len(theta)
    ans = 0.0

    # print('theta')
    # print_array(theta)
    # print('X')
    # print_array(X)

    for i in range(m):
        ans += theta[i] * X[i]

    return ans

def cost(y: list, theta: list, hypo: list) -> float:
    m = len(y)
    ans = 0.0

    # print('Hypothesis')
    # print_array(hypo)
    # print('y')
    # print_array(y)

    for i in range(m):
        ans += ((hypo[i] - y[i]) ** 2)
    
    ans /= (2 * m)
    return ans

def apply(X: list, y: list, num_iter=100) -> list:
    m = len(X[0])
    theta = [0] * (m + 1)
    hypo = [0] * len(y)
    alpha = 0.00000005
    min_cost = float('inf')

    # print('theta in apply')
    # print_array(theta)

    # Add bias
    for x in X:
        x.insert(0, 1)
    for i in range(m):
        hypo[i] = hypothesis(theta, X[i])

    iter = 0

    while True:
        cur_cost = cost(y, theta, hypo)

        
        
        # f.write('iteration #: ' + str(iter) + '\n')
        # f.write('min cost: ' + str(min_cost) + '\n')
        # f.write('current cost: ' + str(cur_cost) + '\n')
        # f.write('\n')

        cur_cost = min(min_cost, cur_cost)
        if min_cost == cur_cost:
            min_theta = theta

        # Change theta using gradient descent
        for j in range(m + 1):
            gradient = 0
            for i in range(len(y)):
                gradient += ((hypo[i] - y[i]) * X[i][j])
            print('gradient: ' + str(gradient))
            theta[j] = theta[j] - ((alpha/m) * gradient)

        iter += 1
        if iter == num_iter:
            break

    return min_theta

def predict(X: list, y: list, theta: list):
    m = len(X)
    hypo = [0] * m

    # Add bias
    for x in X:
        x.insert(0, 1)
    for i in range(m):
        hypo[i] = hypothesis(theta, X[i])

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


def get_redwine_dir():
    dirname = os.path.dirname(os.path.dirname(__file__))
    datasets = os.path.join(dirname, 'datasets')
    redwine_dir = os.path.join(datasets, 'redwine')
    return redwine_dir

def get_dataset_path():
    redwine_dir = get_redwine_dir()
    redwine = os.path.join(redwine_dir, 'winequality-red.csv')
    return redwine

def get_debug_path():
    redwine_dir = get_redwine_dir()
    proj_debug = os.path.join(redwine_dir, 'debug.txt')
    return proj_debug
    
#     # print(os.getcwd())
#     run(redwine)