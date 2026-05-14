__all__ = ["get_train_strategy", "Metrics", "VLAMetrics"]


def __getattr__(name: str):
    if name == "get_train_strategy":
        from .materialize import get_train_strategy

        return get_train_strategy
    if name in {"Metrics", "VLAMetrics"}:
        from .metrics import Metrics, VLAMetrics

        return {"Metrics": Metrics, "VLAMetrics": VLAMetrics}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
