def hypothesis(theta: list, X: list) -> float:
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

def apply(X: list, y: list) -> list:
    m = len(X)
    theta = [0] * (m + 1)
    hypo = [0] * m
    alpha = 0.05
    min_cost = float('inf')

    # Add bias
    X = [x.append(0, 1) for x in X]
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