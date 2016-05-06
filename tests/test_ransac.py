import rh_aligner.common.ransac as R
import rh_renderer.models
import numpy as np
import scipy.stats
import unittest

class TestEnumerateChoices(unittest.TestCase):
    def test_01_k_one(self):
        np.testing.assert_array_equal(R.enumerate_choices(10, 1),
                                      np.arange(10).reshape(10, 1))
    def test_02_k_two(self):
        result = R.enumerate_choices(10, 2)
        a = np.column_stack(((np.arange(100) / 10).astype(int),
                             np.arange(100) % 10))
        a = a[a[:, 0] < a[:, 1]]
        np.testing.assert_array_equal(result, a)
        
    def test_03_k_three(self):
        result = R.enumerate_choices(10, 3)
        a = np.column_stack((
            (np.arange(1000) / 100).astype(int),
            (np.arange(1000) / 10).astype(int) % 10,
            np.arange(1000) % 10))
        a = a[(a[:, 0] < a[:, 1]) & (a[:, 1] < a[:, 2])]
        np.testing.assert_array_equal(result, a)
        
class TestChooseForwardDense(unittest.TestCase):
    def test_01_choose_3(self):
        for _ in range(100):
            a = R.choose_forward_dense(15, 3, 100)
            order = np.lexsort((a[:, 2], a[:, 1], a[:, 0]))
            a = a[order]
            self.assertFalse(np.any(np.all(a[:-1] == a[1:], 1)))
            
    def test_01_choose_4(self):
        for _ in range(100):
            a = R.choose_forward_dense(10, 4, 100)
            order = np.lexsort((a[:, 3], a[:, 2], a[:, 1], a[:, 0]))
            a = a[order]
            self.assertFalse(np.any(np.all(a[:-1] == a[1:], 1)))
            
class TestChooseForwardSparse(unittest.TestCase):
    def test_01_choose_3(self):
        for _ in range(100):
            a = R.choose_forward_sparse(15, 3, 70)
            order = np.lexsort((a[:, 2], a[:, 1], a[:, 0]))
            a = a[order]
            self.assertFalse(np.any(np.all(a[:-1] == a[1:], 1)))
            
    def test_01_choose_4(self):
        for _ in range(100):
            a = R.choose_forward_sparse(10, 4, 70)
            order = np.lexsort((a[:, 3], a[:, 2], a[:, 1], a[:, 0]))
            a = a[order]
            self.assertFalse(np.any(np.all(a[:-1] == a[1:], 1)))

class TestFilterTriangles(unittest.TestCase):
    def test_01_filter_positive(self):
        m0 = np.array([[[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]]] * 6)\
            .reshape(18, 2)
        m1 = np.array([[[1.0, 2.0], [4.0, 2.0], [1.0, 6.0]], # translation
                       [[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]], # rotation
                       [[0.0, 0.0], [3.0 * .9, 0.0], [0.0, 4.0 * .9]], # shrink
                       [[0.0, 0.0], [3.0 * 1.1, 0.0], [0.0, 4.0 * 1.1]], # stretch
                       [[0.0, 0.0], [3.0 * .9, 0.0], [0.0, 4.0]], # distort
                       [[0.0, 0.0], [3.0 * 1.1, 0.0], [0.0, 4.0]], # distort
                       ]).reshape(18, 2)
        choices = np.arange(18).reshape(6, 3)
        result = R.filter_triangles(
            m0, m1, choices, .25, .3)
        np.testing.assert_array_equal(choices, result)
        
    def test_02_filter_negative(self):
        m0 = np.array([[[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]]] * 5)\
            .reshape(15, 2)
        m1 = np.array([[[3.0, 0.0], [0.0, 4.0], [0.0, 0.0]], # complex eigenvalue
                       [[0.0, 0.0], [3.0 * .9, 0.0], [0.0, 4.0 * .9]], # shrink
                       [[0.0, 0.0], [3.0 * 1.1, 0.0], [0.0, 4.0 * 1.1]], # stretch
                       [[0.0, 0.0], [3.0 * .9, 0.0], [0.0, 4.0]], # distort
                       [[0.0, 0.0], [3.0 * 1.1, 0.0], [0.0, 4.0]], # distort
                       ]).reshape(15, 2)
        choices = np.arange(15).reshape(5, 3)
        result = R.filter_triangles(
            m0, m1, choices, .099, .15)
        self.assertEqual(len(result), 0)
    
class TestRansac(unittest.TestCase):
    def test_01_ransac(self):
        #
        # The synthetic alignment
        #
        # x1 = xd + x0 * cos(t) - y0 * sin(t)
        # y1 = yd + x0 * sin(t) + y0 * cos(t)
        #
        r = np.random.RandomState(1010)
        for _ in range(10):
            xd = r.uniform() * 10 - 5
            yd = r.uniform() * 10 - 5
            theta = (r.uniform() - .5) * np.pi / 100
            matrix = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta), np.cos(theta)]])
            #
            # good tuples
            #
            good0 = r.uniform(size=(30, 2)) * 1000 + 1000
            good1 = np.sum(good0[:, :, np.newaxis] *
                           matrix[np.newaxis, :, :], 2) +\
                r.uniform(size=good0.shape) * 5
            good1[:, 0] += xd
            good1[:, 1] += yd
            #
            # bad tuples
            #
            bad0 = r.uniform(size=(5, 2)) * 1000 + 1000
            bad1 = r.uniform(size=(5, 2)) * 1000 + 1000
            m0 = np.vstack((good0, bad0))
            m1 = np.vstack((good1, bad1))
            result = R.ransac(np.array([m0, m1]), 3, 100, 30, .1, 10)
            in_model, model, distances = result
            self.assertTrue(np.all(in_model[:len(good0)]))
            self.assertLess(np.max(np.abs(model.apply(good0) - good1)), 30)

if __name__ == "__main__":
    unittest.main()
