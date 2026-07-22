"""
BAAI/bge-m3 backends via FlagEmbedding.

bge-m3 has three heads on one backbone — dense, sparse (lexical), and
multi-vector (colbert) — and its encode() can return any subset in a single
forward pass. Each head is exposed as a plain single-kind embedder, and
BGEM3FusedEmbedder produces several heads at once: the engine swaps it in
automatically when multiple entries point at the same bge_m3 model and input
column, so declaring dense + sparse + multivector entries costs ONE model load
and ONE forward pass instead of three.
"""

import asyncio
import logging
import threading

from nova_embed.embedders.backends.device import detect_device
from nova_embed.embedders.base import Embedder, FusedEmbedder, OutputKind
from nova_embed.models import Embedding, MultiVectorEmbedding, SparseEmbedding
from nova_embed.registry import EMBEDDERS, FUSED_EMBEDDERS

logger = logging.getLogger(__name__)

# dense head and per-token colbert head both output the backbone width, 1024
_DIMENSIONS = 1024
_NATIVE_MAX_TOKENS = 8192


class _BGEM3Model:
    """
    Shared loader + multi-head encode. The plain and fused embedders both
    delegate here, so head selection lives in exactly one place.
    """

    def __init__(
        self,
        model: str = "BAAI/bge-m3",
        batch_size: int = 32,
        device: str | None = None,
        dtype: str = "float32",
        max_tokens: int | None = None,
    ):
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as e:
            raise RuntimeError(
                "FlagEmbedding is required for the bge_m3 embedders. "
                "Install with: uv add FlagEmbedding"
            ) from e

        self.device = device or detect_device()
        if dtype not in ("float16", "float32"):
            logger.warning(
                "bge_m3 supports dtype 'float16' or 'float32'; got %r — "
                "falling back to float32",
                dtype,
            )
        use_fp16 = dtype == "float16"

        logger.info(
            "Loading %s on %s (dtype=%s)",
            model,
            self.device,
            "float16" if use_fp16 else "float32",
        )
        self._model = BGEM3FlagModel(model, use_fp16=use_fp16, device=self.device)
        # Serializes concurrent pipeline workers (the design contract of
        # pipeline.num_workers): FlagEmbedding's encode is not thread-safe —
        # two threads in one forward pass deadlock on mps.
        self._encode_lock = threading.Lock()
        self.model_name = model
        self.batch_size = batch_size
        self.max_tokens = (
            min(_NATIVE_MAX_TOKENS, max_tokens) if max_tokens else _NATIVE_MAX_TOKENS
        )

    def encode(
        self, texts: list[str], kinds: frozenset[OutputKind]
    ) -> dict[OutputKind, list[Embedding]]:
        with self._encode_lock:
            output = self._model.encode(
                texts,
                batch_size=self.batch_size,
                max_length=self.max_tokens,
                return_dense=OutputKind.DENSE in kinds,
                return_sparse=OutputKind.SPARSE in kinds,
                return_colbert_vecs=OutputKind.MULTIVECTOR in kinds,
            )
        result: dict[OutputKind, list[Embedding]] = {}
        if OutputKind.DENSE in kinds:
            # (N, 1024) ndarray; keep rows as ndarray views, not .tolist() —
            # see the sentence_transformer note on Python-float bloat
            result[OutputKind.DENSE] = list(output["dense_vecs"])
        if OutputKind.SPARSE in kinds:
            # lexical_weights: one {token_id: weight} mapping per input
            result[OutputKind.SPARSE] = [
                SparseEmbedding(
                    indices=[int(token) for token in weights],
                    values=[float(w) for w in weights.values()],
                )
                for weights in output["lexical_weights"]
            ]
        if OutputKind.MULTIVECTOR in kinds:
            # list[np.ndarray] with shape (num_tokens_i, 1024). keep as ndarray
            # — pooling uses np.asarray (no-op on ndarray) and pyarrow writes
            # ndarray directly to list<list<float32>>
            result[OutputKind.MULTIVECTOR] = [
                MultiVectorEmbedding(vectors=vecs) for vecs in output["colbert_vecs"]
            ]
        return result


class _BGEM3Single(Embedder):
    """Common plumbing for the plain single-head bge_m3 embedders."""

    def __init__(self, **kwargs):
        self._core = _BGEM3Model(**kwargs)

    @property
    def model_name(self) -> str:
        return self._core.model_name

    @property
    def max_tokens(self) -> int:
        return self._core.max_tokens

    def _encode(self, texts: list[str]) -> list[Embedding]:
        return self._core.encode(texts, frozenset({self.output_kind}))[
            self.output_kind
        ]

    async def embed(self, texts: list[str]) -> list[Embedding]:
        return await asyncio.to_thread(self._encode, texts)


@EMBEDDERS.register("bge_m3")
class BGEM3DenseEmbedder(_BGEM3Single):
    output_kind = OutputKind.DENSE

    @property
    def dimensions(self) -> int:
        return _DIMENSIONS


@EMBEDDERS.register("bge_m3")
class BGEM3SparseEmbedder(_BGEM3Single):
    output_kind = OutputKind.SPARSE


@EMBEDDERS.register("bge_m3")
class BGEM3MultiVectorEmbedder(_BGEM3Single):
    output_kind = OutputKind.MULTIVECTOR

    @property
    def dimensions(self) -> int:
        return _DIMENSIONS


@FUSED_EMBEDDERS.register("bge_m3")
class BGEM3FusedEmbedder(FusedEmbedder):
    """The requested subset of bge-m3's heads, all from one forward pass."""

    fusable_kinds = frozenset(
        {OutputKind.DENSE, OutputKind.SPARSE, OutputKind.MULTIVECTOR}
    )

    def __init__(self, kinds: frozenset[OutputKind], **kwargs):
        self._kinds = frozenset(kinds)
        self._core = _BGEM3Model(**kwargs)

    @property
    def model_name(self) -> str:
        return self._core.model_name

    @property
    def max_tokens(self) -> int:
        return self._core.max_tokens

    def dimensions_for(self, kind: OutputKind) -> int | None:
        if kind in (OutputKind.DENSE, OutputKind.MULTIVECTOR):
            return _DIMENSIONS
        return None

    async def embed(self, texts: list[str]) -> dict[OutputKind, list[Embedding]]:
        return await asyncio.to_thread(self._core.encode, texts, self._kinds)
