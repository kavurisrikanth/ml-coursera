import unittest
from . import LinearRegression

class TestLinearRegression(unittest.TestCase):
    self,X = [[1, 2, 3], [1, 1, 1], [0, 7, 4]]
    self.y = [6, 3, 11]

    def test_linear(self):
        self.assertEqual(LinearRegression.hypothesis(X, y), 0)

if __name__ == '__main__':
    TestLinearRegression.test_linear()