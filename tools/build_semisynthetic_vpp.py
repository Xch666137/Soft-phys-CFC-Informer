import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physformer.pipelines.semisynthetic_vpp import build_semisynthetic_vpp_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Build the NextGen-based semi-synthetic VPP household and aggregate tables.",
    )
    parser.add_argument(
        "--nextgen-dir",
        type=str,
        default="data_raw/nextgen",
        help="Directory containing downloaded NextGen household CSV files.",
    )
    parser.add_argument(
        "--act-weather-csv",
        type=str,
        default="data_raw/era5/act_canberra_hourly.csv",
        help="Standardized ACT ERA5 hourly weather CSV.",
    )
    parser.add_argument(
        "--rye-generation-csv",
        type=str,
        default="data_raw/rye/rye_generation_and_load.csv",
        help="Rye generation CSV used as the wind-output template source.",
    )
    parser.add_argument(
        "--rye-weather-csv",
        type=str,
        default="data_raw/era5/rye_template_hourly.csv",
        help="Standardized hourly weather CSV aligned to the Rye template site.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data_processed",
        help="Directory for the processed semi-synthetic VPP outputs.",
    )
    parser.add_argument(
        "--wind-penetration-target",
        type=float,
        default=0.15,
        help="Target annual synthetic wind energy share relative to aggregate load energy.",
    )
    parser.add_argument(
        "--source-timezone",
        type=str,
        default="Australia/Sydney",
        help="Timezone used to localize raw NextGen timestamps before converting to UTC.",
    )
    args = parser.parse_args()

    outputs = build_semisynthetic_vpp_dataset(
        nextgen_dir=args.nextgen_dir,
        act_weather_csv=args.act_weather_csv,
        rye_generation_csv=args.rye_generation_csv,
        rye_weather_csv=args.rye_weather_csv,
        output_dir=args.output_dir,
        target_penetration=args.wind_penetration_target,
        source_timezone=args.source_timezone,
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
