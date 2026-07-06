"""
Singleton CLIP model loader.
Provides encode_text() and encode_image() returning normalized numpy float32 vectors.

Multilingual support (M-CLIP):
  If `multilingual-clip` and `transformers` are installed, encode_text() automatically
  uses M-CLIP/XLM-Roberta-Large-Vit-B-32, which supports Chinese, Japanese, Korean,
  and 50+ other languages while remaining fully compatible with the existing 512-D
  FAISS index (no re-indexing required).

No UI, No File IO.
"""

from __future__ import annotations
import threading
from typing import Optional

import numpy as np
from PIL import Image


# ------------------------------------------------------------------ #
#  Lazy imports to avoid hard startup errors if CLIP not installed
# ------------------------------------------------------------------ #

def _import_clip():
    try:
        import clip  # noqa: F401
        import torch
        return clip, torch
    except ImportError as e:
        raise ImportError(
            "OpenAI CLIP is not installed. "
            "Run: pip install git+https://github.com/openai/CLIP.git"
        ) from e


def _import_mclip():
    """
    Try to import multilingual-clip and transformers.
    Returns (multilingual_clip_module, transformers_module) or (None, None).
    """
    try:
        import multilingual_clip  # noqa: F401
        import transformers        # noqa: F401
        return multilingual_clip, transformers
    except ImportError:
        return None, None


# ------------------------------------------------------------------ #
#  Singleton
# ------------------------------------------------------------------ #

_lock = threading.Lock()
_instance: Optional["CLIPModel"] = None


class CLIPModel:
    """
    Thin wrapper around OpenAI CLIP with lazy singleton loading.

    Text encoding strategy (auto-selected at init time):
      • M-CLIP available  → multilingual text encoder (50+ languages incl. Chinese)
      • M-CLIP missing    → standard CLIP text encoder (English-primary)

    Image encoding always uses the standard CLIP ViT-B/32 visual encoder.
    Both paths produce 512-D L2-normalized float32 vectors fully compatible
    with the existing FAISS indices.
    """

    DIM = 512          # ViT-B/32 output dimension
    _MCLIP_MODEL = "M-CLIP/XLM-Roberta-Large-Vit-B-32"

    def __init__(self, model_name: str = "ViT-B/32"):
        clip, torch = _import_clip()
        self._clip = clip
        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load CLIP image (and fallback text) encoder
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

        # Try to load M-CLIP text encoder for multilingual support
        mclip_mod, transformers_mod = _import_mclip()
        if mclip_mod is not None and transformers_mod is not None:
            try:
                self._mclip_model = mclip_mod.load_model(
                    self._MCLIP_MODEL, self.device
                )
                self._mclip_tokenizer = transformers_mod.AutoTokenizer.from_pretrained(
                    self._MCLIP_MODEL
                )
                self._multilingual = True
            except Exception:
                # M-CLIP load failed (e.g. no internet for model download);
                # fall back silently to standard CLIP text encoder.
                self._mclip_model = None
                self._mclip_tokenizer = None
                self._multilingual = False
        else:
            self._mclip_model = None
            self._mclip_tokenizer = None
            self._multilingual = False

    # ---------------------------------------------------------------- #
    #  Public properties
    # ---------------------------------------------------------------- #

    @property
    def multilingual(self) -> bool:
        """True if M-CLIP text encoder is active (Chinese / multilingual search)."""
        return self._multilingual

    # ---------------------------------------------------------------- #
    #  Encoding API
    # ---------------------------------------------------------------- #

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a text string → normalized float32 vector of shape (DIM,).

        Uses M-CLIP when available (supports Chinese and 50+ languages),
        otherwise falls back to standard CLIP (English-primary).
        """
        if self._multilingual:
            return self._encode_text_mclip(text)
        return self._encode_text_clip(text)

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """
        Encode a PIL Image → normalized float32 vector of shape (DIM,).
        Returns a zero vector if encoding fails.
        """
        torch = self._torch
        try:
            tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model.encode_image(tensor)
                features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().astype(np.float32)[0]
        except Exception:
            return np.zeros(self.DIM, dtype=np.float32)

    def encode_image_from_path(self, path: str) -> Optional[np.ndarray]:
        """
        Load an image from disk and encode it.
        Returns None on read failure.
        """
        try:
            img = Image.open(path).convert("RGB")
            return self.encode_image(img)
        except Exception:
            return None

    # ---------------------------------------------------------------- #
    #  Internal — text encoding implementations
    # ---------------------------------------------------------------- #

    def _encode_text_clip(self, text: str) -> np.ndarray:
        """Standard CLIP text encoding (English-primary)."""
        clip, torch = self._clip, self._torch
        with torch.no_grad():
            tokens = clip.tokenize([text], truncate=True).to(self.device)
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)[0]

    def _encode_text_mclip(self, text: str) -> np.ndarray:
        """
        M-CLIP multilingual text encoding.
        Produces a 512-D vector compatible with CLIP image embeddings.
        """
        try:
            embeddings = self._mclip_model.forward(
                [text], self._mclip_tokenizer
            )
            vec = embeddings[0].detach().cpu().numpy().astype(np.float32)
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 1e-8 else vec
        except Exception:
            # Graceful degradation: fall back to standard CLIP on any error
            return self._encode_text_clip(text)


# ------------------------------------------------------------------ #
#  Module-level accessor
# ------------------------------------------------------------------ #

def get_clip_model(model_name: str = "ViT-B/32") -> CLIPModel:
    """Return the singleton CLIP model instance (thread-safe lazy init)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = CLIPModel(model_name)
    return _instance


def is_loaded() -> bool:
    return _instance is not None
