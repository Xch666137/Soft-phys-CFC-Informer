import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physformer.pipelines.semisynthetic_vpp import fetch_rye


def main():
    parser = argparse.ArgumentParser(
        description="Download the public Rye microgrid files used as the wind generation template source.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data_raw/rye",
        help="Directory to store the downloaded Rye files.",
    )
    parser.add_argument(
        "--include-weather-h5",
        action="store_true",
        help="Also download met_data.h5. The default workflow does not require it.",
    )
    args = parser.parse_args()

    downloaded = fetch_rye(args.output_dir, include_weather_h5=args.include_weather_h5)
    print(json.dumps({"downloaded_files": downloaded, "count": len(downloaded)}, indent=2))


if __name__ == "__main__":
    main()
