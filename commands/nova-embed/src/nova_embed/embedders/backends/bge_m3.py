import asyncio
import logging

from nova_embed.embedders.backends.device import detect_device
from nova_embed.embedders.base import Embedder, OutputKind
from nova_embed.models import MultiVectorEmbedding
from nova_embed.registry import EMBEDDERS

logger = logging.getLogger(__name__)


@EMBEDDERS.register("bge_m3")
class BGEM3MultiVectorEmbedder(Embedder):
    """
    Multi-vector (ColBERT-style) output from BAAI/bge-m3 via FlagEmbedding.

    bge-m3 has three heads -- dense, sparse (lexical), and multi-vector (colbert).
    This class only exposes the multi-vector head. If you want dense + sparse
    from the same model, use the (future) hybrid path; for now they are separate
    entries.
    """

    output_kind = OutputKind.MULTIVECTOR

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
                "FlagEmbedding is required for the bge_m3 multivector embedder. "
                "Install with: uv add FlagEmbedding"
            ) from e

        self._device = device or detect_device()
        use_fp16 = dtype == "float16"

        logger.info(
            "Loading %s (multi-vector mode) on %s (dtype=%s)",
            model,
            self._device,
            dtype,
        )
        self._model = BGEM3FlagModel(
            model,
            use_fp16=use_fp16,
            device=self._device,
        )
        self._model_name = model
        self._batch_size = batch_size

        # bge-m3 native max is 8192; allow user to clamp lower
        native_max = 8192
        self._max_tokens = min(native_max, max_tokens) if max_tokens else native_max
        # per-vector dimension is known: colbert head of bge-m3 outputs 1024
        self._dimensions = 1024

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def _encode(self, texts: list[str]) -> list[MultiVectorEmbedding]:
        output = self._model.encode(
            texts,
            batch_size=self._batch_size,
            max_length=self._max_tokens,
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True,
        )
        # output["colbert_vecs"] is list[np.ndarray] with shape (num_tokens_i, 1024).
        # keep as ndarray -- pooling uses np.asarray (no-op on ndarray) and pyarrow
        # writes ndarray directly to list<list<float32>>. avoids a wasteful
        # ndarray -> list -> ndarray round-trip on the pooling path.
        return [MultiVectorEmbedding(vectors=vecs) for vecs in output["colbert_vecs"]]

    async def embed(self, texts: list[str]) -> list[MultiVectorEmbedding]:
        return await asyncio.to_thread(self._encode, texts)
