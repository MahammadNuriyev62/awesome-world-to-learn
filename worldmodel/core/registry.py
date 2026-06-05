"""Game registry and loader.

Games register themselves with ``@register("name")``. The trainer can then load
a game by:

  - registered name:   ``ball``
  - ``gym:`` prefix:    ``gym:ALE/Breakout-v5``  (wrapped by GymGame)
  - file path:          ``./mygame.py``          (imported, first registered or
                                                  Game-like class is used)
  - dotted path:        ``pkg.module:ClassName``

All forms resolve to a ``GameSpec`` carrying a factory ``(config) -> Game``.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import sys
from dataclasses import dataclass, field
from typing import Callable

from .contract import Game

_REGISTRY: dict[str, type] = {}


def register(name: str) -> Callable[[type], type]:
    """Class decorator that registers a game class under ``name``."""

    def deco(cls: type) -> type:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"game name '{name}' already registered to {_REGISTRY[name]}")
        _REGISTRY[name] = cls
        cls.name = getattr(cls, "name", name)
        return cls

    return deco


def registered_names() -> list[str]:
    return sorted(_REGISTRY)


@dataclass
class GameSpec:
    factory: Callable[[dict], Game]
    default_config: dict = field(default_factory=dict)
    display_name: str = "game"

    def make(self, config: dict | None = None) -> Game:
        return self.factory(config or {})


def _default_config_of(cls: type) -> dict:
    fn = getattr(cls, "default_config", None)
    if fn is None:
        return {}
    cfg = fn() if isinstance(inspect.getattr_static(cls, "default_config"), (staticmethod, classmethod)) else fn(cls)
    return dict(cfg or {})


def _spec_from_class(cls: type, display_name: str, bound: dict | None = None) -> GameSpec:
    bound = bound or {}

    def factory(config: dict) -> Game:
        merged = {**bound, **(config or {})}
        try:
            return cls(merged)
        except TypeError:
            # Class does not take a config dict; fall back to a no-arg ctor.
            return cls()

    return GameSpec(factory=factory, default_config=_default_config_of(cls), display_name=display_name)


def _import_path(path: str):
    """Import a standalone .py file as a module and return it."""
    abspath = os.path.abspath(path)
    mod_name = "worldmodel_usergame_" + os.path.splitext(os.path.basename(abspath))[0]
    spec = importlib.util.spec_from_file_location(mod_name, abspath)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import game file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _find_game_class(module) -> type:
    """Pick a Game-like class from a freshly imported module."""
    candidates = [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if obj.__module__ == module.__name__
        and hasattr(obj, "reset")
        and hasattr(obj, "step")
    ]
    if not candidates:
        raise ImportError(f"no Game-like class found in {module.__name__}")
    # Prefer one that is registered (has a .name set by @register).
    for c in candidates:
        if any(c is v for v in _REGISTRY.values()):
            return c
    return candidates[0]


def load_game(spec: str) -> GameSpec:
    """Resolve a game spec string into a GameSpec."""
    # Make sure built-in games are registered.
    import worldmodel.games  # noqa: F401

    if spec.startswith("gym:"):
        from worldmodel.games.gym_adapter import GymGame

        env_id = spec[len("gym:"):]
        return _spec_from_class(GymGame, display_name=f"gym:{env_id}", bound={"env_id": env_id})

    # Dotted path with explicit class: pkg.module:ClassName
    if ":" in spec and not spec.endswith(".py") and "/" not in spec.split(":", 1)[0]:
        mod_name, cls_name = spec.split(":", 1)
        module = importlib.import_module(mod_name)
        cls = getattr(module, cls_name)
        return _spec_from_class(cls, display_name=spec)

    # File path
    if spec.endswith(".py") or os.path.sep in spec or spec.startswith("."):
        module = _import_path(spec)
        cls = _find_game_class(module)
        return _spec_from_class(cls, display_name=os.path.basename(spec))

    # Registered name
    if spec in _REGISTRY:
        return _spec_from_class(_REGISTRY[spec], display_name=spec)

    raise KeyError(
        f"unknown game '{spec}'. Registered: {registered_names()}. "
        "Use a registered name, a gym:ENV_ID, a ./file.py, or a module:Class path."
    )
