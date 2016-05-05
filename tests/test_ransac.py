import rh_aligner.common.ransac as R
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
            


if __name__ == "__main__":
    unittest.main()
