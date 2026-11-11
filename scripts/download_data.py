"""Download the miRBench AGO2_CLASH_Hejret2023 files from Zenodo record 20540907.

    python scripts/download_data.py            # training file only
    python scripts/download_data.py --test     # also the official test file

The test file is for the single final evaluation, after every model choice is
fixed. Do not download it earlier.
"""
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mirtar.data import DATA_DIR, TEST_FILE, TRAIN_FILE, ZENODO_URL  # noqa: E402

SHA256 = {TRAIN_FILE: "9125a81913383e360c1fd73e8096175f247de7c56ea3fc76f141a8b1880f1d91"}


def fetch(name: str) -> None:
    path = DATA_DIR / name
    if path.exists():
        print(f"{path} already exists")
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(f"{ZENODO_URL}{name}?download=1", path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if name in SHA256 and digest != SHA256[name]:
        raise SystemExit(f"{name}: unexpected checksum {digest} (the file on Zenodo may have changed)")
    print(f"{name}: sha256 {digest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", action="store_true", help="also download the official test file (final evaluation only)")
    arguments = parser.parse_args()
    fetch(TRAIN_FILE)
    if arguments.test:
        fetch(TEST_FILE)
