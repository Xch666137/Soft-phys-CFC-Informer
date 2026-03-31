import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physformer.pipelines.semisynthetic_vpp import fetch_nextgen


def main():
    parser = argparse.ArgumentParser(
        description="Download the public NextGen Zenodo household CSV files into a raw data directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data_raw/nextgen",
        help="Directory to store the downloaded NextGen files.",
    )
    args = parser.parse_args()

    downloaded = fetch_nextgen(args.output_dir)
    print(json.dumps({"downloaded_files": downloaded, "count": len(downloaded)}, indent=2))


if __name__ == "__main__":
    main()
