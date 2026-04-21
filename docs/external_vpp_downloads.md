# External VPP Dataset Downloads

This project keeps the thesis training pipeline separate from external-data
acquisition. The download workflow is intentionally lightweight and auditable.

## Goals

- create one canonical local root for external datasets
- record source metadata before any training integration
- fetch public metadata immediately
- download a small set of public HF supplement files for Stage A rehearsal
- prepare restricted-access sources for later application / manual approval

## Local layout

Downloads live under:

```text
data_raw/external/
```

Expected subdirectories:

- `ptprosumer/`
- `ausgrid/`
- `pecan_street/`
- `hemstoec/`
- `hf/`

Each source directory should contain:

- `source_manifest.json`
- `download_notes.md`

## Download script

Use:

```bash
python tools/fetch_external_vpp_sources.py --output-root data_raw/external
```

Initialize directories only:

```bash
python tools/fetch_external_vpp_sources.py --output-root data_raw/external
```

Fetch public metadata and HF parquet listings:

```bash
python tools/fetch_external_vpp_sources.py \
  --output-root data_raw/external \
  --download-public
```

Also download one parquet shard per HF supplement dataset:

```bash
python tools/fetch_external_vpp_sources.py \
  --output-root data_raw/external \
  --download-public \
  --download-hf-parquet \
  --max-hf-files 1
```

## Current source split

- `PTProsumer`
  - public, OSF-backed
  - metadata inspection first
- `Ausgrid`
  - public, landing-page driven
  - landing pages and notes first
- `Pecan Street`
  - restricted access
  - prepare signup, license, and application notes
- `HEMStoEC`
  - public metadata via Zenodo
  - inspect record and small files first
- `HF supplements`
  - public
  - used for Stage A rehearsal, not as thesis main source replacement

## Notes

- The script is conservative by default.
- It does not automatically mirror very large public corpora unless explicitly
  directed to do so.
- The immediate success criterion is having a reproducible local record of where
  each source comes from and what was downloaded.
