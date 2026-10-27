import unittest

import torch
import torch.nn.functional as F

from proof_quality.model import preference_loss


class PreferenceLossTests(unittest.TestCase):
    def test_none_leaves_bradley_terry_loss_unchanged(self) -> None:
        chosen = torch.tensor([1.0, 2.0])
        rejected = torch.tensor([0.0, -1.0])
        loss, accuracy, _ = preference_loss(chosen, rejected, None)
        expected = -F.logsigmoid(chosen - rejected).mean()
        self.assertTrue(torch.allclose(loss, expected))
        self.assertEqual(accuracy.item(), 1.0)

    def test_centering_penalizes_a_shifted_pair(self) -> None:
        chosen = torch.tensor([1.0, 3.0])
        rejected = torch.tensor([-1.0, -3.0])
        coefficient = 0.01
        centered, _, centered_term = preference_loss(chosen, rejected, coefficient)
        shifted, _, shifted_term = preference_loss(
            chosen + 2, rejected + 2, coefficient
        )
        self.assertAlmostEqual(centered_term.item(), 0.0)
        self.assertAlmostEqual(shifted_term.item(), 16.0)
        self.assertAlmostEqual((shifted - centered).item(), coefficient * 16.0)

    def test_zero_coefficient_adds_nothing(self) -> None:
        chosen = torch.tensor([4.0, 4.0])
        rejected = torch.tensor([1.0, 1.0])
        disabled, _, _ = preference_loss(chosen, rejected, None)
        zero, _, _ = preference_loss(chosen, rejected, 0.0)
        self.assertTrue(torch.allclose(disabled, zero))


if __name__ == "__main__":
    unittest.main()
