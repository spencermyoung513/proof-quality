import unittest

from proof_quality.helpers import get_holdout_indices


class HoldoutTests(unittest.TestCase):
    def test_a_group_stays_on_one_side(self) -> None:
        group_ids = [pr for pr in range(30) for _ in range(pr % 3 + 1)]
        keep, holdout = get_holdout_indices(group_ids, fraction=0.1, seed=1998)
        self.assertTrue(set(keep).isdisjoint(holdout))
        self.assertEqual(sorted(keep + holdout), list(range(len(group_ids))))
        self.assertEqual(keep, sorted(keep))
        self.assertEqual(holdout, sorted(holdout))
        keep_groups = {group_ids[index] for index in keep}
        holdout_groups = {group_ids[index] for index in holdout}
        self.assertTrue(keep_groups.isdisjoint(holdout_groups))
        self.assertLess(abs(len(holdout) / len(group_ids) - 0.1), 0.05)
        self.assertEqual(
            get_holdout_indices(group_ids, fraction=0.1, seed=1998),
            get_holdout_indices(group_ids, fraction=0.1, seed=1998),
        )


if __name__ == "__main__":
    unittest.main()
