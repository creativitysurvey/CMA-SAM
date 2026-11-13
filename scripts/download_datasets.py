#!/usr/bin/env python3
"""
Automated dataset downloader & extractor for CMA-SAM:
- Kvasir-SEG (Simula official archive)
- ISIC 2018 Task 1 Lesion Boundary Segmentation (ISIC official S3)
- HAM10000 Images & Lesion Segmentation Masks (Harvard Dataverse DOI: 10.7910/DVN/DBW86T)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import urllib.request
import zipfile


def run_dl(url: str, out_file: pathlib.Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("aria2c"):
        subprocess.run(
            [
                "aria2c", "-x", "16", "-s", "16", "-c",
                "--max-tries=0", "--retry-wait=5",
                "-d", str(out_file.parent), "-o", out_file.name, url,
            ],
            check=True,
        )
    else:
        subprocess.run(["curl", "-L", "-C", "-", "-o", str(out_file), url], check=True)


def setup_kvasir(data_root: pathlib.Path) -> None:
    kv_dir = data_root / "Kvasir-SEG"
    if (kv_dir / "images").exists() and (kv_dir / "masks").exists():
        print("[SKIP] Kvasir-SEG already prepared.")
        return
    zp = data_root / "kvasir-seg.zip"
    run_dl("https://datasets.simula.no/downloads/kvasir-seg.zip", zp)
    with zipfile.ZipFile(zp, "r") as zf:
        zf.extractall(data_root)
    print("[OK] Extracted Kvasir-SEG.")


def setup_isic2018(data_root: pathlib.Path) -> None:
    isic_dir = data_root / "ISIC2018"
    if (isic_dir / "ISIC2018_Task1-2_Training_Input").exists():
        print("[SKIP] ISIC 2018 already prepared.")
        return
    isic_dir.mkdir(parents=True, exist_ok=True)
    inp_zip = isic_dir / "input.zip"
    gt_zip = isic_dir / "gt.zip"
    run_dl("https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Training_Input.zip", inp_zip)
    run_dl("https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Training_GroundTruth.zip", gt_zip)
    for zp in (inp_zip, gt_zip):
        with zipfile.ZipFile(zp, "r") as zf:
            zf.extractall(isic_dir)
    print("[OK] Extracted ISIC 2018.")


def setup_ham10000(data_root: pathlib.Path) -> None:
    ham_dir = data_root / "HAM10000"
    img_dir = ham_dir / "images"
    msk_dir = ham_dir / "masks"
    if img_dir.exists() and msk_dir.exists() and any(img_dir.iterdir()) and any(msk_dir.iterdir()):
        print("[SKIP] HAM10000 already prepared.")
        return
    dl_dir = ham_dir / "downloads"
    for d in (dl_dir, img_dir, msk_dir):
        d.mkdir(parents=True, exist_ok=True)

    api_url = "https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/DBW86T"
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        meta = json.loads(resp.read().decode("utf-8"))

    targets = {
        "HAM10000_images_part_1.zip",
        "HAM10000_images_part_2.zip",
        "HAM10000_segmentations_lesion_tschandl.zip",
    }
    for entry in meta["data"]["latestVersion"]["files"]:
        df = entry["dataFile"]
        if df["filename"] in targets:
            run_dl(f"https://dataverse.harvard.edu/api/access/datafile/{df['id']}", dl_dir / df["filename"])

    for zname, dst in [
        ("HAM10000_images_part_1.zip", img_dir),
        ("HAM10000_images_part_2.zip", img_dir),
        ("HAM10000_segmentations_lesion_tschandl.zip", msk_dir),
    ]:
        with zipfile.ZipFile(dl_dir / zname, "r") as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                base = pathlib.Path(member.filename).name
                if base and not base.startswith("."):
                    (dst / base).write_bytes(zf.read(member))
    print("[OK] Extracted HAM10000.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CMA-SAM medical segmentation benchmarks")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--datasets", nargs="+", default=["kvasir", "isic2018"], choices=["kvasir", "isic2018", "ham10000", "all"])
    args = parser.parse_args()

    root = pathlib.Path(args.data_dir)
    selected = {"kvasir", "isic2018", "ham10000"} if "all" in args.datasets else set(args.datasets)
    if "kvasir" in selected:
        setup_kvasir(root)
    if "isic2018" in selected:
        setup_isic2018(root)
    if "ham10000" in selected:
        setup_ham10000(root)


if __name__ == "__main__":
    main()
