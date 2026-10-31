import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from lightning_nowcasting.data import normalize_weather, read_bundle, read_json
from lightning_nowcasting.inference import make_model, load_model, predict, sha256
from lightning_nowcasting.metrics import occurrence_metrics
from lightning_nowcasting.model import SpectralConv2d
from scripts.make_synthetic_sample import generate
from predict import main as predict_main
from train import main as train_main

ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = read_json(ROOT / "configs/demo.json")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.sample = generate(self.folder / "sample.npz")

    def tearDown(self): self.temp.cleanup()

    def test_corrected_model_identity(self):
        expected = read_json(ROOT / "configs/corrected_bce.json")["model_source_sha256"]
        self.assertEqual(sha256(ROOT / "lightning_nowcasting/model.py"), expected)

    def test_paper_parameter_count(self):
        # Count full published dimensions without allocating hundreds of MB of weights.
        with torch.device("meta"):
            model = make_model(read_json(ROOT / "configs/corrected_bce.json"))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 77546753)

    def test_exporter_cache_reader_is_restricted_and_nonmutating(self):
        from scripts.export_hpc_assets import safe_cache
        path = self.folder / "cache.pt"
        original = {"era5": torch.zeros((4, 8, 3, 3)), "lightning": torch.zeros((4, 3, 3)),
                    "time_index": np.arange("2026-01-01", "2026-01-05", dtype="datetime64[D]").astype("datetime64[ns]"),
                    "lat_vals": np.array([3., 2., 1.]), "lon_vals": np.array([1., 2., 3.])}
        torch.save(original, path); before = sha256(path)
        loaded = safe_cache(path)
        loaded["era5"][0, 0, 0, 0] = 99
        self.assertEqual(sha256(path), before)
        np.testing.assert_array_equal(loaded["time_index"], original["time_index"])

    def test_sample_dimensions_and_order(self):
        d = read_bundle(self.sample, self.config, require_target=True)
        self.assertEqual(d["x"].shape, (2, 9, 3, 17, 19))
        self.assertEqual(d["y"].shape, (2, 1, 17, 19))

    def test_time_gap_rejected(self):
        with np.load(self.sample, allow_pickle=False) as d: values = {k: d[k] for k in d.files}
        values["input_times"][0, 0] -= np.timedelta64(3, "h")
        bad = self.folder / "gap.npz"; np.savez_compressed(bad, **values)
        with self.assertRaisesRegex(ValueError, "consecutive"): read_bundle(bad, self.config)

    def test_channel_swap_rejected(self):
        with np.load(self.sample, allow_pickle=False) as d: values = {k: d[k] for k in d.files}
        values["channel_names"][[0, 1]] = values["channel_names"][[1, 0]]
        bad = self.folder / "channel.npz"; np.savez_compressed(bad, **values)
        with self.assertRaisesRegex(ValueError, "Channel order"): read_bundle(bad, self.config)

    def test_missing_weather_is_explicit(self):
        x = np.ones((1, 8, 3, 4, 5), np.float32); x[0, 0, 0, 0, 0] = np.nan
        mean = np.arange(8, dtype=np.float32) + 1; scale = mean + 1
        with self.assertRaisesRegex(ValueError, "Missing weather"): normalize_weather(x, mean, scale)
        y = normalize_weather(x, mean, scale, missing_policy="legacy_raw_zero")
        self.assertEqual(float(y[0, 0, 0, 0, 0]), -.5)
        np.testing.assert_array_equal(y[0, 1], np.full((3, 4, 5), (1 - 2) / 3, np.float32))

    def test_physical_preparation_end_to_end(self):
        from prepare_input import main as prepare_main
        config = read_json(ROOT / "configs/corrected_bce.json")
        norms = read_json(ROOT / "configs/normalization.json")
        weather = np.broadcast_to(np.asarray(norms["mean"], np.float32)[None, :, None, None, None],
                                   (1, 8, 3, 129, 135)).copy()
        times = np.array([["2026-01-01T00", "2026-01-01T01", "2026-01-01T02"]], dtype="datetime64[ns]")
        raw = self.folder / "raw.npz"; prepared = self.folder / "prepared.npz"
        np.savez_compressed(raw, weather=weather, lightning_history=np.zeros((1, 1, 3, 129, 135), np.uint8),
                            latitude=np.asarray(norms["latitude"]), longitude=np.asarray(norms["longitude"]),
                            input_times=times, target_times=np.array(["2026-01-01T03"], dtype="datetime64[ns]"))
        with patch("sys.argv", ["prepare_input.py", "--raw", str(raw), "--output", str(prepared)]): prepare_main()
        data = read_bundle(prepared, config)
        np.testing.assert_array_equal(data["x"], np.zeros((1, 9, 3, 129, 135), np.float32))

    def test_mode_specific_weights_and_gradients(self):
        # An impulse in Fourier mode (1,2) must use only that mode's channel matrix.
        layer = SpectralConv2d(2, 3, 3)
        with torch.no_grad():
            layer.weights_positive.zero_(); layer.weights_negative.zero_()
            layer.weights_positive[:, :, 1, 2] = torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=torch.cfloat)
        spectrum = torch.zeros((1, 2, 8, 5), dtype=torch.cfloat)
        spectrum[0, :, 1, 2] = torch.tensor([2., 3.], dtype=torch.cfloat)
        x = torch.fft.irfft2(spectrum, s=(8, 8))
        expected_spectrum = torch.zeros((1, 3, 8, 5), dtype=torch.cfloat)
        expected_spectrum[0, :, 1, 2] = torch.tensor([14., 19., 24.], dtype=torch.cfloat)
        expected = torch.fft.irfft2(expected_spectrum, s=(8, 8))
        actual = layer(x)
        torch.testing.assert_close(actual, expected)
        actual.square().mean().backward()
        self.assertTrue(torch.isfinite(layer.weights_positive.grad).all())

    def test_negative_band_is_used(self):
        layer = SpectralConv2d(1, 1, 2)
        with torch.no_grad():
            layer.weights_positive.zero_(); layer.weights_negative.zero_()
            layer.weights_negative[0, 0, 1, 1] = 2
        spectrum = torch.zeros((1, 1, 8, 5), dtype=torch.cfloat); spectrum[0, 0, -1, 1] = 3
        x = torch.fft.irfft2(spectrum, s=(8, 8))
        expected = torch.fft.irfft2(spectrum * 2, s=(8, 8))
        torch.testing.assert_close(layer(x), expected)

    def test_threshold_is_strict_and_metrics_match_counts(self):
        p = np.array([.3, .8, .7, .1], np.float32); y = np.array([1, 1, 0, 0])
        r = occurrence_metrics(p, y, .3)
        self.assertEqual(r["counts"], {"TP": 1, "TN": 1, "FP": 1, "FN": 1})
        self.assertAlmostEqual(r["metrics"]["CSI"], 1 / 3)
        self.assertEqual(r["metrics"]["POD"], .5)
        self.assertEqual(r["metrics"]["MCC"], 0)
        self.assertEqual(r["metrics"]["Pattern_Correlation"], occurrence_metrics(p, y, .6)["metrics"]["Pattern_Correlation"])

    def test_no_checkpoint_silently_becomes_random_model(self):
        with self.assertRaisesRegex(ValueError, "checkpoint is required"):
            load_model(self.config, None, torch.device("cpu"))

    def test_checkpoint_identity_guard(self):
        bad = self.folder / "bad.pt"; bad.write_bytes(b"wrong checkpoint")
        config = copy.deepcopy(self.config); config["checkpoint_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256"):
            load_model(config, bad, torch.device("cpu"))

    def test_demo_cli_end_to_end_and_no_overwrite(self):
        out = self.folder / "predictions"
        args = ["--config", str(ROOT / "configs/demo.json"), "--input", str(self.sample),
                "--output", str(out), "--device", "cpu", "--demo"]
        report = predict_main(args)
        self.assertTrue(report["untrained_synthetic_demo"])
        with np.load(out / "prediction.npz", allow_pickle=False) as d:
            self.assertEqual(d["probability"].shape, (2, 1, 17, 19))
            np.testing.assert_array_equal(d["binary"], (d["probability"] > np.float32(.3)).astype(np.uint8))
        with self.assertRaises(FileExistsError): predict_main(args)

    def test_short_training_and_checkpoint_reload(self):
        val = generate(self.folder / "validation.npz", start="2026-02-01T00")
        out = self.folder / "trained"
        config = train_main(["--config", str(ROOT / "configs/demo.json"), "--train", str(self.sample),
                             "--validation", str(val), "--output", str(out), "--epochs", "2", "--device", "cpu"])
        model, digest = load_model(config, out / "best.pt", torch.device("cpu"))
        self.assertEqual(digest, config["checkpoint_sha256"])
        data = read_bundle(val, config)
        self.assertTrue(np.isfinite(predict(model, data["x"], torch.device("cpu"))).all())
        self.assertFalse(json.loads((out / "training_summary.json").read_text())["test_evaluated"])

    def test_training_rejects_shared_hours(self):
        with self.assertRaisesRegex(ValueError, "share input/target hours"):
            train_main(["--config", str(ROOT / "configs/demo.json"), "--train", str(self.sample),
                        "--validation", str(self.sample), "--output", str(self.folder / "bad_training"), "--epochs", "1"])


if __name__ == "__main__": unittest.main()
