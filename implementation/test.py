import unittest
import pandas as pd
from LinearRegression import naive, helper

class TestLinearRegression(unittest.TestCase):
    def __init__(self):
        self.X = [[1, 2, 3], [1, 1, 1], [0, 7, 4]]
        self.y = [6, 3, 11]

    def test_linear(self):
        theta = [0, 0, 0]

        hypo = []
        for x in self.X:
            h = naive.hypothesis(x, theta)
            self.assertEqual(h, 0)
            hypo.append(h)
        
        c = naive.cost(self.y, theta, hypo)
        self.assertAlmostEqual(c, 166/6)

def go():
    df = pd.read_csv(helper.get_dataset_path(), header=0)
    print(df.head())

if __name__ == '__main__':
    # test = TestLinearRegression()
    # test.test_linear()

    # LinearRegression.go()
    go()