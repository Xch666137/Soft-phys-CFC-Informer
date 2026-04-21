import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


PECAN_STREET_SIGNUP_URL = "https://dataport.pecanstreet.org/"
PECAN_STREET_ACCESS_URL = "https://www.pecanstreet.org/access/"
PECAN_STREET_ABOUT_URL = "https://www.pecanstreet.org/dataport/about-dataport/"
PECAN_STREET_LICENSE_URL = "https://www.pecanstreet.org/dataport/licenses/"

PTPROSUMER_OSF_URL = "https://osf.io/9xs3a/"
PTPROSUMER_OSF_API = "https://api.osf.io/v2/nodes/9xs3a/files/osfstorage/"
PTPROSUMER_PAPER_URL = "https://www.nature.com/articles/s41597-025-06118-x"

HEMSTOEC_ZENODO_URL = "https://zenodo.org/records/8096648"
HEMSTOEC_ZENODO_API = "https://zenodo.org/api/records/8096648"
HEMSTOEC_PAPER_URL = "https://www.nature.com/articles/s41597-024-03184-5"

AUSGRID_RESOURCE_URL = "https://data.gov.au/data/en/dataset/nsw-solar-home-electricty-data/resource/d2dc76f0-22e3-4efc-bed9-bb4e0e50f0db"
AUSGRID_LANDING_URL = "https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data"

HF_DATASETS = {
    "tulipa762_electricity_load_diagrams": {
        "repo_id": "tulipa762/electricity_load_diagrams",
        "source_url": "https://hf.co/datasets/tulipa762/electricity_load_diagrams",
        "time_span": "2011-2014",
        "resolution": "1h",
        "entity_unit": "client",
        "native_signals": ["load"],
    },
    "weijie1996_load_timeseries": {
        "repo_id": "Weijie1996/load_timeseries",
        "source_url": "https://hf.co/datasets/Weijie1996/load_timeseries",
        "time_span": "multi-year, mixed by source",
        "resolution": "15m/30m/1h",
        "entity_unit": "household",
        "native_signals": ["load"],
    },
}


SOURCE_CATALOG = {
    "ptprosumer": {
        "source_name": "PTProsumer",
        "source_url": PTPROSUMER_OSF_URL,
        "license_access": "public (OSF)",
        "time_span": "2018-2024",
        "resolution": "1s net-load, 1min irradiance",
        "entity_unit": "prosumer",
        "native_signals": ["net", "pv", "irradiance"],
        "notes": [
            f"Paper: {PTPROSUMER_PAPER_URL}",
            f"OSF page: {PTPROSUMER_OSF_URL}",
            "Public source; inspect file listing and sizes before deciding whether to mirror everything locally.",
        ],
    },
    "ausgrid": {
        "source_name": "Ausgrid Solar Home Electricity Data",
        "source_url": AUSGRID_RESOURCE_URL,
        "license_access": "public (Data.NSW / Ausgrid)",
        "time_span": "2010-07-01 to 2013-06-30",
        "resolution": "30min",
        "entity_unit": "solar home",
        "native_signals": ["load", "pv"],
        "notes": [
            f"Data.NSW resource: {AUSGRID_RESOURCE_URL}",
            f"Ausgrid landing page: {AUSGRID_LANDING_URL}",
            "Public source; main task is to capture landing links and verify the downloadable CSV package.",
        ],
    },
    "pecan_street": {
        "source_name": "Pecan Street Dataport",
        "source_url": PECAN_STREET_SIGNUP_URL,
        "license_access": "restricted / signup / university or commercial license",
        "time_span": "since 2009",
        "resolution": "sub-hourly, high frequency",
        "entity_unit": "home / circuit",
        "native_signals": ["load", "pv", "battery", "ev", "water", "net"],
        "notes": [
            f"Signup: {PECAN_STREET_SIGNUP_URL}",
            f"Access info: {PECAN_STREET_ACCESS_URL}",
            f"About: {PECAN_STREET_ABOUT_URL}",
            f"Licenses: {PECAN_STREET_LICENSE_URL}",
            "Record university signup requirements, approval status, and target fields for the thesis.",
        ],
    },
    "hemstoec": {
        "source_name": "HEMStoEC",
        "source_url": HEMSTOEC_ZENODO_URL,
        "license_access": "public metadata; dataset availability via Zenodo record",
        "time_span": "2020-01 to 2023-02",
        "resolution": "household time series",
        "entity_unit": "household",
        "native_signals": ["load", "pv", "battery", "weather"],
        "notes": [
            f"Zenodo record: {HEMSTOEC_ZENODO_URL}",
            f"Paper: {HEMSTOEC_PAPER_URL}",
            "Check the record files and confirm whether battery/SOC variables are directly recoverable.",
        ],
    },
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def fetch_json(url: str) -> dict[str, Any]:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.json()


def download_file(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> None:
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)


def download_text(url: str, destination: Path) -> None:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    destination.write_text(response.text, encoding="utf-8")


def build_manifest(base: dict[str, Any], directory: Path, status: str) -> dict[str, Any]:
    return {
        "source_name": base["source_name"],
        "source_url": base["source_url"],
        "license_access": base["license_access"],
        "download_date": iso_now(),
        "time_span": base["time_span"],
        "resolution": base["resolution"],
        "entity_unit": base["entity_unit"],
        "native_signals": base["native_signals"],
        "local_directory": str(directory),
        "status": status,
    }


def build_notes(base: dict[str, Any], extra_lines: list[str] | None = None) -> str:
    lines = [
        f"# {base['source_name']}",
        "",
        f"- Source URL: {base['source_url']}",
        f"- License / Access: {base['license_access']}",
        f"- Expected time span: {base['time_span']}",
        f"- Expected resolution: {base['resolution']}",
        f"- Entity unit: {base['entity_unit']}",
        f"- Native signals: {', '.join(base['native_signals'])}",
        "",
        "## Notes",
    ]
    lines.extend([f"- {line}" for line in base.get("notes", [])])
    if extra_lines:
        lines.append("")
        lines.append("## Download Actions")
        lines.extend([f"- {line}" for line in extra_lines])
    return "\n".join(lines) + "\n"


def init_source(root: Path, key: str) -> Path:
    directory = ensure_dir(root / key)
    meta = SOURCE_CATALOG[key]
    write_json(directory / "source_manifest.json", build_manifest(meta, directory, status="initialized"))
    write_text(directory / "download_notes.md", build_notes(meta))
    return directory


def inspect_ptprosumer(directory: Path) -> list[str]:
    actions = []
    payload = fetch_json(PTPROSUMER_OSF_API)
    write_json(directory / "osf_file_listing.json", payload)
    actions.append("Saved OSF file listing to osf_file_listing.json.")
    data_entries = payload.get("data", [])
    if data_entries:
        actions.append(f"Root listing returned {len(data_entries)} entries.")
    else:
        actions.append("Root listing returned no direct entries; inspect links in osf_file_listing.json.")
    meta = SOURCE_CATALOG["ptprosumer"]
    write_json(directory / "source_manifest.json", build_manifest(meta, directory, status="inspected"))
    write_text(directory / "download_notes.md", build_notes(meta, actions))
    return actions


def inspect_ausgrid(directory: Path) -> list[str]:
    actions = []
    status = "inspected"
    try:
        download_text(AUSGRID_RESOURCE_URL, directory / "data_gov_au_resource.html")
        actions.append("Saved Data.gov.au resource page HTML.")
    except Exception as exc:  # pragma: no cover - network dependent
        status = "partial"
        actions.append(f"Failed to fetch Data.gov.au resource page: {exc}")
    try:
        download_text(AUSGRID_LANDING_URL, directory / "ausgrid_landing.html")
        actions.append("Saved Ausgrid landing page HTML.")
    except Exception as exc:  # pragma: no cover - network dependent
        status = "partial"
        actions.append(f"Failed to fetch Ausgrid landing page: {exc}")
    meta = SOURCE_CATALOG["ausgrid"]
    write_json(directory / "source_manifest.json", build_manifest(meta, directory, status=status))
    write_text(directory / "download_notes.md", build_notes(meta, actions))
    return actions


def inspect_pecan_street(directory: Path) -> list[str]:
    actions = []
    status = "application_prepared"
    for name, url in {
        "signup.html": PECAN_STREET_SIGNUP_URL,
        "access.html": PECAN_STREET_ACCESS_URL,
        "about.html": PECAN_STREET_ABOUT_URL,
        "licenses.html": PECAN_STREET_LICENSE_URL,
    }.items():
        try:
            download_text(url, directory / name)
            actions.append(f"Saved {name}.")
        except Exception as exc:  # pragma: no cover - network dependent
            status = "partial"
            actions.append(f"Failed to fetch {name}: {exc}")
    actions.append("University signup requires institutional verification and a short research description.")
    meta = SOURCE_CATALOG["pecan_street"]
    write_json(directory / "source_manifest.json", build_manifest(meta, directory, status=status))
    write_text(directory / "download_notes.md", build_notes(meta, actions))
    return actions


def inspect_hemstoec(directory: Path, download_small_files: bool) -> list[str]:
    actions = []
    payload = fetch_json(HEMSTOEC_ZENODO_API)
    write_json(directory / "zenodo_record.json", payload)
    actions.append("Saved Zenodo record metadata.")
    downloaded_files = []
    if download_small_files:
        for file_info in payload.get("files", []):
            size = int(file_info.get("size", 0) or 0)
            if size > 25 * 1024 * 1024:
                continue
            url = file_info.get("links", {}).get("self") or file_info.get("links", {}).get("download")
            key = file_info.get("key")
            if url and key:
                destination = directory / safe_name(key)
                if not destination.exists():
                    download_file(url, destination)
                downloaded_files.append(destination.name)
    if downloaded_files:
        actions.append(f"Downloaded small public files: {', '.join(downloaded_files)}")
    else:
        actions.append("No small public files were downloaded automatically; inspect zenodo_record.json for the full file list.")
    meta = SOURCE_CATALOG["hemstoec"]
    write_json(directory / "source_manifest.json", build_manifest(meta, directory, status="inspected"))
    write_text(directory / "download_notes.md", build_notes(meta, actions))
    return actions


def inspect_hf_dataset(root: Path, key: str, repo_id: str, source_meta: dict[str, Any], download_parquet: bool, max_files: int) -> list[str]:
    directory = ensure_dir(root / "hf" / key)
    actions = []
    parquet_api = f"https://datasets-server.huggingface.co/parquet?dataset={repo_id}"
    hub_api = f"https://huggingface.co/api/datasets/{repo_id}"
    parquet_files: list[dict[str, Any]] = []
    status = "inspected"
    try:
        payload = fetch_json(parquet_api)
        write_json(directory / "parquet_listing.json", payload)
        actions.append("Saved HF parquet listing.")
        parquet_files = payload.get("parquet_files", [])
    except Exception as exc:  # pragma: no cover - network dependent
        status = "partial"
        actions.append(f"Parquet API unavailable: {exc}")
        repo_payload = fetch_json(hub_api)
        write_json(directory / "repo_info.json", repo_payload)
        actions.append("Saved HF dataset repo metadata from hub API.")
        try:
            download_text(f"https://huggingface.co/datasets/{repo_id}/resolve/main/README.md", directory / "README.md")
            actions.append("Saved HF dataset README.")
        except Exception as readme_exc:  # pragma: no cover - network dependent
            actions.append(f"Failed to fetch HF dataset README: {readme_exc}")
    downloaded = []
    if download_parquet:
        for file_info in parquet_files[:max_files]:
            url = file_info.get("url")
            filename = file_info.get("filename")
            if not url or not filename:
                continue
            destination = directory / filename
            if not destination.exists():
                download_file(url, destination)
            downloaded.append(filename)
    if downloaded:
        actions.append(f"Downloaded parquet shards: {', '.join(downloaded)}")
    elif parquet_files:
        actions.append("Parquet listing is available but shards were not downloaded in this run.")
    else:
        actions.append("No parquet shards were listed by the datasets server.")

    manifest = {
        "source_name": f"Hugging Face dataset: {repo_id}",
        "source_url": source_meta["source_url"],
        "license_access": "public (HF Hub)",
        "download_date": iso_now(),
        "time_span": source_meta["time_span"],
        "resolution": source_meta["resolution"],
        "entity_unit": source_meta["entity_unit"],
        "native_signals": source_meta["native_signals"],
        "local_directory": str(directory),
        "status": "downloaded" if downloaded else status,
    }
    write_json(directory / "source_manifest.json", manifest)
    notes = [
        f"HF repo: {repo_id}",
        f"Parquet API: {parquet_api}",
        f"Expected time span: {source_meta['time_span']}",
        f"Expected resolution: {source_meta['resolution']}",
    ] + actions
    write_text(directory / "download_notes.md", "\n".join([f"# HF Supplement: {repo_id}", ""] + [f"- {line}" for line in notes]) + "\n")
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize and inspect external VPP data sources for cross-year training.",
    )
    parser.add_argument("--output-root", default="data_raw/external", help="Root directory for external-source assets.")
    parser.add_argument(
        "--sources",
        nargs="*",
        default=["ptprosumer", "ausgrid", "pecan_street", "hemstoec", "hf"],
        help="Subset of sources to process. Available: ptprosumer, ausgrid, pecan_street, hemstoec, hf.",
    )
    parser.add_argument(
        "--download-public",
        action="store_true",
        help="Fetch public metadata and safe downloadable assets where available.",
    )
    parser.add_argument(
        "--download-hf-parquet",
        action="store_true",
        help="Download parquet shards for the configured HF supplemental datasets.",
    )
    parser.add_argument(
        "--max-hf-files",
        type=int,
        default=1,
        help="Maximum number of parquet shards to download per HF dataset.",
    )
    parser.add_argument(
        "--download-small-zenodo-files",
        action="store_true",
        help="For Zenodo-backed sources, also download files smaller than 25 MB.",
    )
    args = parser.parse_args()

    output_root = ensure_dir(Path(args.output_root))
    actions_summary: dict[str, Any] = {}

    def run_step(name: str, fn: Any) -> None:
        try:
            actions_summary[name] = fn()
        except Exception as exc:  # pragma: no cover - network/path dependent
            actions_summary[name] = {
                "status": "error",
                "error": str(exc),
            }

    for key in ["ptprosumer", "ausgrid", "pecan_street", "hemstoec"]:
        if key in args.sources:
            init_source(output_root, key)

    if "ptprosumer" in args.sources and args.download_public:
        run_step("ptprosumer", lambda: inspect_ptprosumer(output_root / "ptprosumer"))

    if "ausgrid" in args.sources and args.download_public:
        run_step("ausgrid", lambda: inspect_ausgrid(output_root / "ausgrid"))

    if "pecan_street" in args.sources and args.download_public:
        run_step("pecan_street", lambda: inspect_pecan_street(output_root / "pecan_street"))

    if "hemstoec" in args.sources and args.download_public:
        run_step(
            "hemstoec",
            lambda: inspect_hemstoec(
                output_root / "hemstoec",
                download_small_files=args.download_small_zenodo_files,
            ),
        )

    if "hf" in args.sources:
        ensure_dir(output_root / "hf")
        for key, meta in HF_DATASETS.items():
            repo_id = meta["repo_id"]
            if args.download_public:
                run_step(
                    key,
                    lambda key=key, repo_id=repo_id, meta=meta: inspect_hf_dataset(
                        output_root,
                        key,
                        repo_id,
                        meta,
                        download_parquet=args.download_hf_parquet,
                        max_files=args.max_hf_files,
                    ),
                )
            else:
                directory = ensure_dir(output_root / "hf" / key)
                manifest = {
                    "source_name": f"Hugging Face dataset: {repo_id}",
                    "source_url": meta["source_url"],
                    "license_access": "public (HF Hub)",
                    "download_date": iso_now(),
                    "time_span": meta["time_span"],
                    "resolution": meta["resolution"],
                    "entity_unit": meta["entity_unit"],
                    "native_signals": meta["native_signals"],
                    "local_directory": str(directory),
                    "status": "initialized",
                }
                write_json(directory / "source_manifest.json", manifest)
                write_text(
                    directory / "download_notes.md",
                    "\n".join(
                        [
                            f"# HF Supplement: {repo_id}",
                            "",
                            f"- Source URL: {meta['source_url']}",
                            f"- Expected time span: {meta['time_span']}",
                            f"- Expected resolution: {meta['resolution']}",
                            "- Run with --download-public to fetch parquet listing and optional shards.",
                        ]
                    )
                    + "\n",
                )

    summary_path = output_root / "download_summary.json"
    write_json(
        summary_path,
        {
            "generated_at": iso_now(),
            "output_root": str(output_root),
            "sources": args.sources,
            "download_public": bool(args.download_public),
            "download_hf_parquet": bool(args.download_hf_parquet),
            "actions": actions_summary,
        },
    )
    print(json.dumps({"output_root": str(output_root), "summary_path": str(summary_path), "actions": actions_summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
