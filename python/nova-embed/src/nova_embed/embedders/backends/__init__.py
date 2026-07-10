"""Concrete embedder backends, laid out flat by backend — NOT by kind or modality.

A backend module may register several classes under the same config `type` name
(e.g. `sentence_transformer` is both a dense and a sparse backend — the registry
keys on (output_kind, type)), and a single class may serve several modalities
(e.g. the ST dense embedder handles text and images for CLIP-family models).
Kind and modality are declarations on the class, not directory structure.

Importing this package runs every @EMBEDDERS.register decorator. A backend
whose heavy dependency isn't installed (torch/sentence-transformers live in the
`embed` extra) is skipped with a debug log — it simply won't appear in the
registry, and selecting it in config fails with the registry's "available
backends" message.
"""

import importlib
import logging

logger = logging.getLogger(__name__)

_BACKEND_MODULES = [
    "sentence_transformer",
    "openai",
    "fastembed",
    "bge_m3",
]

for _mod in _BACKEND_MODULES:
    try:
        importlib.import_module(f"nova_embed.embedders.backends.{_mod}")
    except ImportError as e:
        logger.debug(
            "embedder backend %r unavailable (%s) — install nova-embed[embed]",
            _mod,
            e,
        )
