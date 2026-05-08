import asyncio
import logging

import torch

from vectorforge.embedders.multivector.base import MultiVectorEmbedder
from vectorforge.models import MultiVectorEmbedding

logger = logging.getLogger(__name__)


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class BGEM3MultiVectorEmbedder(MultiVectorEmbedder):
    """
    Multi-vector (ColBERT-style) output from BAAI/bge-m3 via FlagEmbedding.

    bge-m3 has three heads -- dense, sparse (lexical), and multi-vector (colbert).
    This class only exposes the multi-vector head. If you want dense + sparse
    from the same model, use the (future) hybrid path; for now they are separate
    configs.
    """

    DTYPE_MAP = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    def __init__(
        self,
        model: str = "BAAI/bge-m3",
        batch_size: int = 32,
        device: str | None = None,
        dtype: str = "float32",
        max_tokens: int | None = None,
        truncate: bool = False,
    ):
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as e:
            raise RuntimeError(
                "FlagEmbedding is required for the bge_m3 multivector embedder. "
                "Install with: uv add FlagEmbedding"
            ) from e

        self._device = device or _detect_device()
        torch_dtype = self.DTYPE_MAP.get(dtype, torch.float32)
        use_fp16 = torch_dtype == torch.float16

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
        self._truncate = truncate

        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model)
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

    def split_text(self, text: str) -> list[str]:
        # truncate mode: emit one piece, let the encoder chop at max_tokens.
        if self._truncate:
            return [text]

        # split mode: tokenize and break into max_tokens-sized pieces; each piece
        # becomes its own multivector record. Useful when you want passage-level
        # multivector coverage of long docs rather than truncating.
        tokens = self._tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= self._max_tokens:
            return [text]
        pieces = []
        for i in range(0, len(tokens), self._max_tokens):
            chunk = tokens[i : i + self._max_tokens]
            pieces.append(self._tokenizer.decode(chunk, skip_special_tokens=True))
        return pieces

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
