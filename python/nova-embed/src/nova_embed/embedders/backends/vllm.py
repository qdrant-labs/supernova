"""vLLM backend: dense embeddings through vLLM's pooling runner.

This is the one backend that can embed text and images TOGETHER (`modality:
multimodal`): vLLM's input format takes a prompt plus optional
`multi_modal_data`, so a single entry can map several source columns into one
vector space (Qwen3-VL-Embedding and friends). It also serves as a plain
high-throughput text embedder (`modality: text`, e.g. Qwen3-Embedding) and an
image-only embedder (`modality: image`).

Canonical inputs, exactly as the engine hands them over:

* ``str``       — text
* ``PIL.Image`` — image
* ``dict``      — multimodal parts ``{"text": str, "image": PIL.Image}``,
                  either key optional (at least one present)

Two prompt paths:

* **raw** — plain strings straight into ``llm.embed()``. Correct for ordinary
  embedding models that were not trained on chat-formatted input.
* **chat template** — the tokenizer's chat template renders a conversation
  (optional `instruction` system turn + user content), and images ride along
  as ``multi_modal_data``. Required for VLM-based embedding models: the
  template is what injects the image placeholder tokens.

The path is picked per batch: anything with an image always goes through the
template; pure text uses the template only when `instruction` is set or
`use_chat_template: true` forces it. `instruction` changes the embedding
space — query-side embedding at search time MUST reproduce it exactly, which
is why it's surfaced via the `instruction` property into the run manifest.
"""

import asyncio
import logging
import threading

from typing import Any

import numpy as np

from vllm import LLM

from nova_embed.embedders.base import Embedder, OutputKind
from nova_embed.media import Modality
from nova_embed.registry import EMBEDDERS

logger = logging.getLogger(__name__)


@EMBEDDERS.register("vllm")
class VLLMEmbedder(Embedder):
    output_kind = OutputKind.DENSE
    supported_modalities = frozenset(
        {Modality.TEXT, Modality.IMAGE, Modality.MULTIMODAL}
    )

    def __init__(
        self,
        model: str,
        # max items per llm.embed() call. None = hand vLLM the whole chunk and
        # let continuous batching schedule it (usually what you want); set it
        # only to bound host memory, e.g. decoded PIL images in flight.
        batch_size: int | None = None,
        dtype: str = "auto",
        trust_remote_code: bool = False,
        instruction: str | None = None,
        use_chat_template: bool | None = None,  # None = auto (see module docs)
        # forwarded to vllm.LLM: gpu_memory_utilization, max_model_len,
        # limit_mm_per_prompt, enforce_eager, ...
        **engine_kwargs: Any,
    ):
        logger.info("Loading %s via vLLM pooling runner (dtype=%s)", model, dtype)
        self._llm = LLM(
            model=model,
            runner="pooling",
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            **engine_kwargs,
        )
        self._tokenizer = self._llm.get_tokenizer()
        self._model_name = model
        self._batch_size = batch_size
        self._instruction = instruction
        self._use_chat_template = use_chat_template
        # llm.embed() is synchronous and not re-entrant; serialize like the
        # other local backends and keep the event loop free via to_thread.
        self._encode_lock = threading.Lock()
        try:
            self._max_tokens = self._llm.llm_engine.model_config.max_model_len
        except AttributeError:
            self._max_tokens = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def max_tokens(self) -> int | None:
        return self._max_tokens

    @property
    def instruction(self) -> str | None:
        return self._instruction

    def _templated_input(self, text: Any, image: Any) -> dict:
        """
        One chat-templated vLLM input: prompt text + optional image data.
        """
        content: list[dict] = []
        if image is not None:
            content.append({"type": "image", "image": image})
        if text:
            content.append({"type": "text", "text": text})
        if not content:
            content.append({"type": "text", "text": ""})

        conversation: list[dict] = []
        if self._instruction:
            conversation.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self._instruction}],
                }
            )
        conversation.append({"role": "user", "content": content})

        prompt = self._tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inp: dict = {"prompt": prompt}
        if image is not None:
            inp["multi_modal_data"] = {"image": image}
        return inp

    def _to_input(self, item: Any) -> dict:
        if isinstance(item, str):
            return self._templated_input(item, None)
        if isinstance(item, dict):  # multimodal parts from the engine
            return self._templated_input(item.get("text"), item.get("image"))
        return self._templated_input(None, item)  # canonical image (PIL)

    def _encode(self, batch: list[Any]) -> list[np.ndarray]:
        raw_text = all(isinstance(x, str) for x in batch)
        use_template = (
            self._use_chat_template
            if self._use_chat_template is not None
            else (not raw_text or self._instruction is not None)
        )
        inputs = [self._to_input(x) for x in batch] if use_template else list(batch)

        embeddings: list[np.ndarray] = []
        with self._encode_lock:
            step = self._batch_size or len(inputs)
            for start in range(0, len(inputs), step):
                outputs = self._llm.embed(
                    inputs[start : start + step], use_tqdm=False
                )
                # float32 ndarray rows, not .tolist() — see the ST backend note
                embeddings.extend(
                    np.asarray(o.outputs.embedding, dtype=np.float32)
                    for o in outputs
                )
        return embeddings

    async def embed(self, batch: list[Any]) -> list[np.ndarray]:
        return await asyncio.to_thread(self._encode, batch)
