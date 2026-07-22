"""Deterministic in-process embedder backends for tests (no ML deps).

Registered under type "fake" for all three kinds — exercising exactly the
property that made the registry key on (kind, type). The same classes are
also registered as "fake_fused", a type that ADDITIONALLY has a fused class
in FUSED_EMBEDDERS: entries with type "fake" always get plain units, entries
with "fake_fused" fuse when they group.
"""

from __future__ import annotations

from nova_embed.embedders.base import Embedder, FusedEmbedder
from nova_embed.media import Modality
from nova_embed.models import MultiVectorEmbedding, OutputKind, SparseEmbedding
from nova_embed.registry import EMBEDDERS, FUSED_EMBEDDERS

# Every FakeDense / FakeFused instantiation lands here so tests can assert
# instance sharing and fusion grouping.
DENSE_INSTANTIATIONS: list["FakeDenseEmbedder"] = []
FUSED_INSTANTIATIONS: list["FakeFusedEmbedder"] = []


@EMBEDDERS.register("fake", "fake_fused")
class FakeDenseEmbedder(Embedder):
    output_kind = OutputKind.DENSE

    def __init__(self, model: str = "fake-dense", dim: int = 2, **_ignored):
        self._model = model
        self._dim = dim
        DENSE_INSTANTIATIONS.append(self)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dim

    @property
    def max_tokens(self) -> int:
        return 512

    async def embed(self, batch):
        # encodes input length -> tests can assert truncation etc.
        return [[float(len(item))] * self._dim for item in batch]


@EMBEDDERS.register("fake", "fake_fused")
class FakeSparseEmbedder(Embedder):
    output_kind = OutputKind.SPARSE

    def __init__(self, model: str = "fake-sparse", **_ignored):
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    async def embed(self, batch):
        return [
            SparseEmbedding(indices=[len(item)], values=[1.0]) for item in batch
        ]


@EMBEDDERS.register("fake", "fake_fused")
class FakeMultiVectorEmbedder(Embedder):
    output_kind = OutputKind.MULTIVECTOR

    def __init__(self, model: str = "fake-mv", **_ignored):
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return 2

    async def embed(self, batch):
        return [
            MultiVectorEmbedding(vectors=[[3.0, 0.0], [0.0, 4.0]]) for _ in batch
        ]


@FUSED_EMBEDDERS.register("fake_fused")
class FakeFusedEmbedder(FusedEmbedder):
    """Produces the same values as the plain fakes, from "one forward pass"."""

    fusable_kinds = frozenset(
        {OutputKind.DENSE, OutputKind.SPARSE, OutputKind.MULTIVECTOR}
    )

    def __init__(
        self,
        kinds: frozenset[OutputKind],
        model: str = "fake-fused",
        dim: int = 2,
        batch_size: int = 32,
        **_ignored,
    ):
        self.kinds = frozenset(kinds)
        self._model = model
        self._dim = dim
        self.batch_size = batch_size
        FUSED_INSTANTIATIONS.append(self)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def max_tokens(self) -> int:
        return 512

    def dimensions_for(self, kind: OutputKind) -> int | None:
        return None if kind == OutputKind.SPARSE else self._dim

    async def embed(self, batch):
        out = {}
        if OutputKind.DENSE in self.kinds:
            out[OutputKind.DENSE] = [[float(len(i))] * self._dim for i in batch]
        if OutputKind.SPARSE in self.kinds:
            out[OutputKind.SPARSE] = [
                SparseEmbedding(indices=[len(i)], values=[1.0]) for i in batch
            ]
        if OutputKind.MULTIVECTOR in self.kinds:
            out[OutputKind.MULTIVECTOR] = [
                MultiVectorEmbedding(vectors=[[3.0, 0.0], [0.0, 4.0]]) for _ in batch
            ]
        return out


@EMBEDDERS.register("fake_image")
class FakeImageEmbedder(Embedder):
    output_kind = OutputKind.DENSE
    supported_modalities = frozenset({Modality.TEXT, Modality.IMAGE})

    def __init__(self, model: str = "fake-image", **_ignored):
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    async def embed(self, batch):
        return [[1.0, 2.0] for _ in batch]


@EMBEDDERS.register("fake_mm")
class FakeMultimodalEmbedder(Embedder):
    """Mirrors the vllm backend's input contract: str, PIL.Image, or a part
    dict {"text": ..., "image": ...} with at least one key present."""

    output_kind = OutputKind.DENSE
    supported_modalities = frozenset(
        {Modality.TEXT, Modality.IMAGE, Modality.MULTIMODAL}
    )

    def __init__(self, model: str = "fake-mm", instruction: str | None = None, **_ignored):
        self._model = model
        self._instruction = instruction

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def instruction(self) -> str | None:
        return self._instruction

    async def embed(self, batch):
        # [len(text), has_image] — tests assert exactly which parts arrived
        out = []
        for item in batch:
            if isinstance(item, dict):
                text, image = item.get("text"), item.get("image")
            elif isinstance(item, str):
                text, image = item, None
            else:
                text, image = None, item
            out.append([float(len(text)) if text else 0.0, 1.0 if image is not None else 0.0])
        return out
