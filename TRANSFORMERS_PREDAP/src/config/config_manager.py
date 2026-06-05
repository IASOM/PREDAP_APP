from typing import Optional
from src.config.base_transformer_config import BaseTransformerConfig

_CONFIG: Optional[BaseTransformerConfig] = None

def set_config(cfg: BaseTransformerConfig):
    global _CONFIG
    _CONFIG = cfg

def get_config() -> BaseTransformerConfig:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = BaseTransformerConfig()
    return _CONFIG
