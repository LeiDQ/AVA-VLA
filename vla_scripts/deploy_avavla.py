"""Compatibility wrapper for `vla-scripts/deploy_avavla.py`."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "vla-scripts" / "deploy_avavla.py"
_SPEC = spec_from_file_location("_avavla_deploy_script", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load AVA-VLA deploy script from {_SCRIPT_PATH}")

_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

load_avavla_model = _MODULE.load_avavla_model
predict_action = _MODULE.predict_action
batch_predict = _MODULE.batch_predict
compute_efficiency_metrics = _MODULE.compute_efficiency_metrics

__all__ = ["load_avavla_model", "predict_action", "batch_predict", "compute_efficiency_metrics"]
