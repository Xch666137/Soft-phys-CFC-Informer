import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physformer.pipelines.semisynthetic_vpp import (
    SITE_REGISTRY,
    enrich_weather_with_cloud_cover,
    fetch_era5_cloud_cover_cds,
)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch ERA5 total cloud cover from the full single-levels dataset and merge it into an existing standardized weather CSV.",
    )
    parser.add_argument("--site-key", type=str, required=True, choices=sorted(SITE_REGISTRY))
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--weather-csv", type=str, required=True)
    parser.add_argument("--cloud-cover-csv", type=str, required=True)
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--raw-download-dir", type=str, default=None)
    args = parser.parse_args()

    cloud_csv = fetch_era5_cloud_cover_cds(
        output_csv=args.cloud_cover_csv,
        start_date=args.start_date,
        end_date=args.end_date,
        site_key=args.site_key,
        raw_download_dir=args.raw_download_dir,
    )
    merged_csv = enrich_weather_with_cloud_cover(
        weather_csv=args.weather_csv,
        cloud_cover_csv=cloud_csv,
        output_csv=args.output_csv,
    )
    print(json.dumps({
        "site_key": args.site_key,
        "cloud_cover_csv": str(cloud_csv),
        "output_csv": str(merged_csv),
    }, indent=2))


if __name__ == "__main__":
    main()
