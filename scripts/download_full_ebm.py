from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

DOWNLOAD_URL = "https://update.kbv.de/ita-update/Stammdateien/SDEBM/SDEBM_V1.61.zip"


def download_zip(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Using existing ZIP file: {destination}")
        return destination

    print(f"Downloading {url} to {destination}...")
    with urllib.request.urlopen(url) as response, destination.open("wb") as out_file:
        shutil.copyfileobj(response, out_file)
    print("Download complete.")
    return destination


def extract_zip(zip_path: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} to {extract_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction complete.")
    return extract_dir


def find_xml_file(extract_dir: Path) -> Path | None:
    xml_files = sorted(extract_dir.rglob("*.xml"))
    if not xml_files:
        return None

    for candidate in xml_files:
        if "850_" in candidate.name:
            return candidate
    return xml_files[0]


def copy_xml(source: Path, destination: Path, force: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        print(f"Using existing XML file: {destination}")
        return destination
    print(f"Copying {source} to {destination}...")
    shutil.copy2(source, destination)
    print("XML copy complete.")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and extract the full KBV EBM dataset.")
    parser.add_argument("--url", default=DOWNLOAD_URL, help="Download URL for the EBM ZIP archive.")
    parser.add_argument("--zip", default="data/SDEBM_V1.61.zip", help="Local path for the downloaded ZIP file.")
    parser.add_argument("--extract", default="data/sdebm_extracted", help="Directory where the archive will be extracted.")
    parser.add_argument("--xml-output", default="data/ebm.xml", help="Path to write the extracted XML file for ingestion.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing downloaded or extracted files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    zip_path = Path(args.zip)
    extract_dir = Path(args.extract)
    xml_output = Path(args.xml_output)

    if args.force and zip_path.exists():
        zip_path.unlink()
    if args.force and extract_dir.exists():
        shutil.rmtree(extract_dir)
    if args.force and xml_output.exists():
        xml_output.unlink()

    try:
        downloaded_zip = download_zip(args.url, zip_path)
        extracted_dir = extract_zip(downloaded_zip, extract_dir)
        xml_file = find_xml_file(extracted_dir)
        if xml_file is None:
            print(f"No XML file found in extracted archive at {extracted_dir}.")
            return 1

        print(f"Found XML file: {xml_file}")
        copy_xml(xml_file, xml_output, force=args.force)
        print(
            f"Full EBM source is ready at {xml_output}.\n"
            "Now run `python scripts/ingest_ebm.py --xml data/ebm.xml --output data/processed` to build artifacts."
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
