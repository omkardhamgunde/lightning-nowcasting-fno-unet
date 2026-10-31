"""Download and safely install the checksum-pinned trained model and real sample."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = {"corrected_bce_epoch3.pt", "real_sample.npz", "expected_prediction.npz",
         "real_sample_manifest.json", "model_config.json", "normalization.json"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def install(archive, output, release):
    archive, output = Path(archive), Path(output).resolve()
    if output.exists(): raise FileExistsError("Output already exists; nothing overwritten: " + str(output))
    if archive.stat().st_size != release["archive_bytes"] or sha256(archive) != release["archive_sha256"]:
        raise ValueError("Downloaded archive size/SHA256 differs from the pinned release")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist()) != len(FILES) or set(z.namelist()) != FILES:
            raise ValueError("Unexpected archive contents")
        for item in z.infolist():
            if Path(item.filename).name != item.filename or ((item.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError("Unsafe archive entry")
        if sum(item.file_size for item in z.infolist()) > 1_000_000_000:
            raise ValueError("Unexpected expanded archive size")
        manifest = json.loads(z.read("real_sample_manifest.json"))
        if manifest["checkpoint_sha256"] != release["checkpoint_sha256"]:
            raise ValueError("Manifest checkpoint differs from the pinned release")
        if set(manifest["asset_sha256"]) != {"corrected_bce_epoch3.pt", "real_sample.npz", "expected_prediction.npz"}:
            raise ValueError("Unexpected sample manifest checksum fields")
        with tempfile.TemporaryDirectory(prefix=".lightning-assets-", dir=output.parent) as temporary:
            staging = Path(temporary).resolve()
            if staging.parent != output.parent: raise ValueError("Temporary folder escaped its parent")
            contents = staging / "contents"; contents.mkdir()
            for name in sorted(FILES):
                with z.open(name) as src, (contents / name).open("xb") as dst: shutil.copyfileobj(src, dst)
            for name, digest in manifest["asset_sha256"].items():
                if sha256(contents / name) != digest: raise ValueError("Extracted asset checksum differs: " + name)
            if sha256(contents / "corrected_bce_epoch3.pt") != release["checkpoint_sha256"]:
                raise ValueError("Trained checkpoint identity differs")
            contents.rename(output)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "sample_assets")
    p.add_argument("--archive", type=Path, help="Install an already downloaded archive without downloading again")
    args = p.parse_args()
    release = json.loads((ROOT / "configs/release_assets.json").read_text(encoding="utf-8"))
    if args.output.exists(): raise FileExistsError("Output already exists; nothing overwritten")
    if args.archive:
        print("Installed:", install(args.archive, args.output, release))
        return
    if not release["archive_url"].startswith("https://github.com/omkardhamgunde/lightning-nowcasting-fno-unet/releases/download/"):
        raise ValueError("Unexpected release host/path")
    with tempfile.TemporaryDirectory(prefix="lightning-download-") as temporary:
        archive = Path(temporary) / "reviewer_sample_assets.zip"
        request = urllib.request.Request(release["archive_url"], headers={"User-Agent": "lightning-nowcasting-sample"})
        print("Downloading the verified model and sample (~572 MB).", flush=True)
        total = 0; next_progress = 32 * 1024 * 1024
        with urllib.request.urlopen(request, timeout=120) as response, archive.open("xb") as stream:
            while True:
                block = response.read(4 * 1024 * 1024)
                if not block: break
                total += len(block)
                if total > release["archive_bytes"]: raise ValueError("Download exceeds the declared archive size")
                stream.write(block)
                if total >= next_progress:
                    print(f"Downloaded {total / 1_000_000:.0f} MB", flush=True)
                    next_progress += 32 * 1024 * 1024
        print("Installed:", install(archive, args.output, release))


if __name__ == "__main__": main()
