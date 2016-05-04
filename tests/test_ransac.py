import rh_aligner.common.ransac as R
import numpy as np
import scipy.stats
import unittest

class TestChooseForward(unittest.TestCase):
    def test_00_none(self):
        a = np.zeros(0, int)
        R.choose_forward(a, 10, 0)

    def test_01_choose_once(self):
        a = -np.ones(10, int)
        R.choose_forward(a, 20, 10)
        self.assertTrue(np.all(a >= 0))
        self.assertTrue(np.all(a < 20))
        self.assertEqual(len(np.unique(a)), 10)
        self.assertSequenceEqual(a.tolist(), sorted(a))

    def test_02_choose_wisely(self):
        # Ensure that the distribution is reasonably random
        #
        a = np.zeros((100, 10), int)
        for idx in range(len(a)):
            R.choose_forward(a[idx], 20, 10)
        counts = np.bincount(a.flatten())
        b = scipy.stats.binom(1000, .05)
        #
        # Make sure that there's less than 1/1000 chance
        # that we'd get a low # (binomial is overestimating what
        # this number should be (~70))
        #
        self.assertGreater(np.min(counts), b.ppf(.001))

if __name__ == "__main__":
    unittest.main()
