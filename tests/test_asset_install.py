import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.download_assets import install, FILES


class AssetInstallTests(unittest.TestCase):
    def make_archive(self, folder, *, extra=None):
        payload = {"corrected_bce_epoch3.pt": b"checkpoint fixture, never an actual model",
                   "real_sample.npz": b"sample fixture", "expected_prediction.npz": b"reference fixture",
                   "model_config.json": b"{}", "normalization.json": b"{}"}
        digest = hashlib.sha256(payload["corrected_bce_epoch3.pt"]).hexdigest()
        payload["real_sample_manifest.json"] = json.dumps({"checkpoint_sha256": digest,
            "asset_sha256": {name: hashlib.sha256(payload[name]).hexdigest() for name in
                             ["corrected_bce_epoch3.pt", "real_sample.npz", "expected_prediction.npz"]}}).encode()
        path = folder / "fixture.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for name, value in payload.items(): z.writestr(name, value)
            if extra: z.writestr(extra, b"not allowed")
        return path, {"archive_bytes": path.stat().st_size,
                      "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "checkpoint_sha256": digest}

    def test_checked_install_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); path, release = self.make_archive(root); output = root / "assets"
            install(path, output, release)
            self.assertEqual({f.name for f in output.iterdir()}, FILES)
            with self.assertRaises(FileExistsError): install(path, output, release)

    def test_bad_digest_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); path, release = self.make_archive(root); release["archive_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA256"): install(path, root / "assets", release)
            self.assertFalse((root / "assets").exists())

    def test_unexpected_path_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); path, release = self.make_archive(root, extra="../outside.txt")
            with self.assertRaisesRegex(ValueError, "contents"): install(path, root / "assets", release)
            self.assertFalse((root / "outside.txt").exists())


if __name__ == "__main__": unittest.main()
