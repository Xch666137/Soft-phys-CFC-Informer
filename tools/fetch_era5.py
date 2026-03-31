import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physformer.pipelines.semisynthetic_vpp import SITE_REGISTRY, fetch_era5_cds


def main():
    parser = argparse.ArgumentParser(
        description="Download hourly ERA5 single-level weather for a registered template site using CDS API.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="Destination CSV path for the standardized weather time series.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Inclusive start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="Inclusive end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--site-key",
        type=str,
        default="act_canberra",
        choices=sorted(SITE_REGISTRY),
        help="Registered site key. Use act_canberra for output weather and rye_template for wind-template fitting.",
    )
    parser.add_argument(
        "--raw-download-path",
        type=str,
        default=None,
        help="Optional path for the raw NetCDF download before CSV normalization.",
    )
    args = parser.parse_args()

    output_csv = fetch_era5_cds(
        output_csv=args.output_csv,
        start_date=args.start_date,
        end_date=args.end_date,
        site_key=args.site_key,
        raw_download_path=args.raw_download_path,
    )
    print(json.dumps({"output_csv": str(output_csv), "site_key": args.site_key}, indent=2))


if __name__ == "__main__":
    main()
