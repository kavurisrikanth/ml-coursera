def hypothesis(theta: list, X: list) -> float:
    '''
    Returns the hypothesis function result for the given theta and X vectors.
    '''
    m = len(theta)
    ans = 0.0

    for i in range(m):
        ans += theta[i] * X[i]

    return ans

def cost(X: list, y: list) -> float:
    m = len(X)
    ans = 0.0
    theta = [0] * m

    for i in range(m):
        ans += ((hypothesis(theta, X[i]) - y[i]) ** 2)
    
    ans /= (2 * m)
    return ans

def apply(X: list, y: list) -> list:
    m = len(X)
    theta = [0] * (m + 1)
    hypo = [0] * m
    alpha = 0.05

    X = [x.append(0, 1) for x in X]
    for i in range(m):
        hypo[i] = hypothesis(theta, X[i])

    # Change theta using gradient descent
    new_theta = [0] * (m + 1)
    for j in range(m + 1):
        gradient = 0
        for i in range(m):
            gradient += ((hypo[i] - y[i]) * X[j][i])
        new_theta[j] = theta[j] - ((alpha/m) * gradient)