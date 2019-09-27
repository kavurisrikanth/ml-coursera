import unittest
import LinearRegression

class TestLinearRegression(unittest.TestCase):
    def __init__(self):
        self.X = [[1, 2, 3], [1, 1, 1], [0, 7, 4]]
        self.y = [6, 3, 11]

    def test_linear(self):
        theta = [0, 0, 0]

        for x in self.X:
            self.assertEqual(LinearRegression.hypothesis(theta, x), 0)

if __name__ == '__main__':
    test = TestLinearRegression()
    test.test_linear()