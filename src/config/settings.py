"""
Application-level settings (persisted to ~/.localmuse/settings.json).
Pure Python — no PyQt5 dependency.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict


_SETTINGS_DIR  = Path.home() / ".localmuse"
_SETTINGS_FILE = _SETTINGS_DIR / "settings.json"

_DEFAULTS: dict = {
    # UI
    "language": "en",
    "last_library_path": "",
    # Modality toggles
    "semantic_enabled":  True,
    "color_enabled":     True,
    "structure_enabled": True,
    # Modality weights
    "semantic_weight":   1.0,
    "color_weight":      0.6,
    "structure_weight":  0.8,
    # Search
    "top_k":         200,
    "thumbnail_size": 360,
}


class Settings:
    """Persistent application settings — pure Python, no Qt."""

    def __init__(self):
        self._data: dict = dict(_DEFAULTS)
        self._load()

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        try:
            if _SETTINGS_FILE.exists():
                raw = json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
                for k, v in raw.items():
                    if k in _DEFAULTS:
                        self._data[k] = v
        except Exception:
            pass

    def save(self) -> None:
        try:
            _SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
            _SETTINGS_FILE.write_text(
                json.dumps(self._data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #  Generic get / set
    # ------------------------------------------------------------------ #

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def set(self, key: str, value) -> None:
        self._data[key] = value
        self.save()

    # ------------------------------------------------------------------ #
    #  Typed properties
    # ------------------------------------------------------------------ #

    @property
    def language(self) -> str:
        return str(self._data.get("language", "en"))

    @language.setter
    def language(self, v: str) -> None:
        self.set("language", v)

    @property
    def last_library_path(self) -> str:
        return str(self._data.get("last_library_path", ""))

    @last_library_path.setter
    def last_library_path(self, v: str) -> None:
        self.set("last_library_path", v)

    # Semantic
    @property
    def semantic_enabled(self) -> bool:
        return bool(self._data.get("semantic_enabled", True))

    @semantic_enabled.setter
    def semantic_enabled(self, v: bool) -> None:
        self.set("semantic_enabled", v)

    @property
    def semantic_weight(self) -> float:
        return float(self._data.get("semantic_weight", 1.0))

    @semantic_weight.setter
    def semantic_weight(self, v: float) -> None:
        self.set("semantic_weight", float(v))

    # Color
    @property
    def color_enabled(self) -> bool:
        return bool(self._data.get("color_enabled", True))

    @color_enabled.setter
    def color_enabled(self, v: bool) -> None:
        self.set("color_enabled", v)

    @property
    def color_weight(self) -> float:
        return float(self._data.get("color_weight", 0.6))

    @color_weight.setter
    def color_weight(self, v: float) -> None:
        self.set("color_weight", float(v))

    # Structure / Sketch
    @property
    def structure_enabled(self) -> bool:
        return bool(self._data.get("structure_enabled", True))

    @structure_enabled.setter
    def structure_enabled(self, v: bool) -> None:
        self.set("structure_enabled", v)

    @property
    def structure_weight(self) -> float:
        return float(self._data.get("structure_weight", 0.8))

    @structure_weight.setter
    def structure_weight(self, v: float) -> None:
        self.set("structure_weight", float(v))

    # Misc
    @property
    def top_k(self) -> int:
        return int(self._data.get("top_k", 200))

    @top_k.setter
    def top_k(self, v: int) -> None:
        self.set("top_k", int(v))

    @property
    def thumbnail_size(self) -> int:
        return int(self._data.get("thumbnail_size", 360))

    # ------------------------------------------------------------------ #
    #  Effective weights (0 if modality disabled, then normalized)
    # ------------------------------------------------------------------ #

    def effective_weights(self) -> Dict[str, float]:
        """Return per-modality weights, normalized over enabled modalities."""
        raw = {
            "semantic":  self.semantic_weight  if self.semantic_enabled  else 0.0,
            "color":     self.color_weight     if self.color_enabled     else 0.0,
            "structure": self.structure_weight if self.structure_enabled else 0.0,
        }
        total = sum(raw.values())
        if total == 0:
            return {"semantic": 1.0, "color": 0.0, "structure": 0.0}
        return {k: v / total for k, v in raw.items()}
