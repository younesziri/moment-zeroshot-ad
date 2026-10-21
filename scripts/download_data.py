"""Download the data of the 248-series study (docs/analysis_plan.md, section 1).

- TSB-UAD-Public (the authors' exact files): only its KDD21 folder, the 250 UCR series, is extracted
  into data/TSB-UAD-Public/KDD21/.
- The original UCR archive (for the sanity checks), kept as a zip in data/ucr/.
Both are gitignored. Files already present are not downloaded again. The SHA-256 of each archive is
printed, so anyone can check they have the same files.

Usage:  uv run python scripts/download_data.py
"""

import hashlib
import urllib.request
import zipfile
from pathlib import Path

TSB_UAD = ("https://www.thedatum.org/datasets/TSB-UAD-Public.zip", Path("data/TSB-UAD-Public/TSB-UAD-Public.zip"))
UCR = ("https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip",
       Path("data/ucr/UCR_TimeSeriesAnomalyDatasets2021.zip"))


def download(url: str, path: Path) -> str:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url} ...")
        # Some servers refuse urllib's default user agent (HTTP 403).
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request) as response:
            path.write_bytes(response.read())
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"{path}: {path.stat().st_size / 1e6:.0f} MB, SHA-256 {sha}")
    return sha


def extract_kdd21(zip_path: Path, target: Path) -> int:
    """Extract the KDD21 .out files only. Downloaded archives are untrusted: refuse any entry whose
    path would land outside the target folder."""
    count = 0
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if "/KDD21/" not in name or not name.endswith(".out"):
                continue
            out = (target / Path(name).name).resolve()
            if target.resolve() not in out.parents:
                raise ValueError(f"unsafe path in archive: {name}")
            out.write_bytes(z.read(name))
            count += 1
    return count


def main():
    download(*TSB_UAD)
    target = Path("data/TSB-UAD-Public/KDD21")
    target.mkdir(parents=True, exist_ok=True)
    print(f"extracted {extract_kdd21(TSB_UAD[1], target)} KDD21 files into {target}")
    download(*UCR)


if __name__ == "__main__":
    main()
