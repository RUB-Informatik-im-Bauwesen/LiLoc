import unittest

import numpy as np

import image_tools


class TestSum(unittest.TestCase):

    def test_simple_transform(self):
        points = np.array([[0, 0], [0, 1], [0.5, 0.5]], dtype=np.float32)
        m1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        t_points = image_transformations.transform_point(points, m1)
        self.assertEqual(points, t_points, "Points should be equal")

    def test_sum_tuple(self):
        self.assertEqual(sum((1, 2, 2)), 6, "Should be 6")


if __name__ == '__main__':
    unittest.main()
