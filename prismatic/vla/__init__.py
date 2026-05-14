def get_vla_dataset_and_collator(*args, **kwargs):
    """Lazily import dataset materialization so model-only imports do not require RLDS/dlimp dependencies."""
    from .materialize import get_vla_dataset_and_collator as _get_vla_dataset_and_collator

    return _get_vla_dataset_and_collator(*args, **kwargs)
