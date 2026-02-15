from __future__ import annotations

from typing import Callable, Dict, Type, Any


_REGISTRY: Dict[str, Dict[str, Any]] = {
    "image_encoder": {},
    "text_encoder": {},
    "projection": {},
    "loss": {},
}


def register(kind: str, name: str):
    def decorator(obj):
        _REGISTRY[kind][name] = obj
        return obj
    return decorator


def get(kind: str, name: str):
    if name not in _REGISTRY[kind]:
        available = ", ".join(sorted(_REGISTRY[kind].keys()))
        raise KeyError(f"Unknown {kind}='{name}'. Available: {available}")
    return _REGISTRY[kind][name]
