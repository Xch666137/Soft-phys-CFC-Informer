from importlib import import_module


__all__ = [
    "build_portfolio_dataset",
    "build_semisynthetic_vpp_dataset",
    "export_portfolio_forecasts",
    "fetch_era5_cds",
    "fetch_nextgen",
    "fetch_rye",
    "summarize_runs",
    "validate_portfolio_forecasts",
]


def __getattr__(name: str):
    if name in {
        "build_portfolio_dataset",
        "export_portfolio_forecasts",
        "summarize_runs",
        "validate_portfolio_forecasts",
    }:
        module = import_module(".thesis_ops", __name__)
        return getattr(module, name)

    if name in {
        "build_semisynthetic_vpp_dataset",
        "fetch_era5_cds",
        "fetch_nextgen",
        "fetch_rye",
    }:
        module = import_module(".semisynthetic_vpp", __name__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
