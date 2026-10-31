import unittest
import numpy as np
from scripts.verify_sample import compare_reference


class ReferenceComparisonTests(unittest.TestCase):
    def setUp(self):
        self.limits = {"maximum_absolute_probability_error": 5e-4,
                       "maximum_mean_probability_error": 5e-6, "require_exact_binary_at_0p30": True}

    def test_small_sparse_backend_difference_is_reported(self):
        ref = np.full((1000,), .5, np.float32); actual = ref.copy(); actual[0] += 2e-4
        report = compare_reference(actual, ref, ref > np.float32(.3), self.limits)
        self.assertEqual(report["binary_differences_at_0p30"], 0)
        self.assertFalse(report["original_strict_allclose"])

    def test_threshold_crossing_is_rejected(self):
        ref = np.full((1000,), .5, np.float32); ref[0] = .30001; actual = ref.copy(); actual[0] = .29999
        with self.assertRaisesRegex(ValueError, "Binary sample predictions differ"):
            compare_reference(actual, ref, ref > np.float32(.3), self.limits)

    def test_large_probability_error_is_rejected(self):
        ref = np.full((1000,), .5, np.float32); actual = ref.copy(); actual[0] += .01
        with self.assertRaisesRegex(ValueError, "tolerance"):
            compare_reference(actual, ref, ref > np.float32(.3), self.limits)

    def test_systematic_probability_error_is_rejected(self):
        ref = np.full((1000,), .5, np.float32); actual = ref + 1e-4
        with self.assertRaisesRegex(ValueError, "tolerance"):
            compare_reference(actual, ref, ref > np.float32(.3), self.limits)

    def test_strict_option_preserves_the_original_gate(self):
        ref = np.full((1000,), .5, np.float32); actual = ref.copy(); actual[0] += 2e-4
        with self.assertRaisesRegex(ValueError, "Strict probability"):
            compare_reference(actual, ref, ref > np.float32(.3), self.limits, strict=True)


if __name__ == "__main__": unittest.main()
