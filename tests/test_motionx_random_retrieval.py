import unittest

import numpy as np
import torch

from tools.audit_motionx_random_retrieval import canonical_order, retrieval


class RandomRetrievalTests(unittest.TestCase):
    def test_perfect_pairs_and_tail(self):
        x = torch.eye(67)
        p = np.random.default_rng(42).permutation(67)
        np.testing.assert_array_equal(retrieval(x, x, p), [1, 1, 1])

    def test_numpy_reference_and_joint_reordering(self):
        rng = np.random.default_rng(5)
        a = rng.normal(size=(95, 16)).astype(np.float32)
        b = rng.normal(size=(95, 16)).astype(np.float32)
        order = rng.permutation(95)
        counts = np.zeros(3)
        for start in range(0, 95, 32):
            ids = order[start:start + 32]
            d = ((a[ids, None, :] - b[ids][None, :, :]) ** 2).sum(-1)
            ranks = np.argsort(d, axis=1)
            hits = ranks[:, :3] == np.arange(len(ids))[:, None]
            counts += hits.cumsum(-1).sum(0)
        np.testing.assert_allclose(retrieval(torch.from_numpy(a), torch.from_numpy(b), order), counts / 95)
        np.testing.assert_allclose(retrieval(torch.from_numpy(a[order]), torch.from_numpy(b[order]), np.arange(95)), counts / 95)

    def test_identity_alignment(self):
        np.testing.assert_array_equal(canonical_order(['c', 'a', 'b'], ['a', 'b', 'c']), [1, 2, 0])
        for ids in (['a', 'a', 'b'], ['a', 'b', 'd']):
            with self.assertRaises(ValueError):
                canonical_order(ids, ['a', 'b', 'c'])

    def test_invalid_inputs(self):
        x = torch.eye(32)
        for order in (np.zeros(32, dtype=int), np.arange(31)):
            with self.assertRaises(ValueError):
                retrieval(x, x, order)
        with self.assertRaises(ValueError):
            retrieval(x, x, np.arange(32), [31])
        x[0, 0] = float('nan')
        with self.assertRaises(ValueError):
            retrieval(x, x, np.arange(32))


if __name__ == '__main__':
    unittest.main()