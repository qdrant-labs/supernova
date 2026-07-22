from abc import ABC, abstractmethod


class Chunker(ABC):
    """Splits a record's text into pieces BEFORE embedding.

    Chunking is deliberately model-agnostic and owned here, not by the embedders
    (see issue #12). Every model in a pipeline receives the SAME pieces, so a
    dense + sparse run can't split one text into different counts. The chunker
    does NOT guarantee token fit: a piece that still exceeds a model's token
    limit is truncated by that model at embed time. Picking a sensible
    strategy/model pairing is the user's responsibility.
    """

    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        """Return one or more pieces for ``text``, in order.

        Must return at least one element — return ``[text]`` for a no-op.
        """
