"""Config schema tests: the launch-time harassment layer.

Every cross-entry invariant (unique names/columns, chunking × multi-input,
modality × chunking, per-entry field rules) must die at model_validate time —
none of these may survive to a running pipeline.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nova_embed.config import EmbedConfig, EmbedderEntry

BASE = {
    "source": {"type": "huggingface_parquet", "dataset_name": "org/data"},
    "storage": {"type": "local", "output_dir": "/tmp/x"},
}


def entry(**overrides) -> dict:
    e = {
        "name": "minilm",
        "kind": "dense",
        "type": "sentence_transformer",
        "model": "org/model",
        "input_column": "text",
        "modality": "text",
    }
    e.update(overrides)
    return e


def config(*entries, **top) -> EmbedConfig:
    return EmbedConfig.model_validate({**BASE, "embedders": list(entries), **top})


def test_minimal_config_parses():
    cfg = config(entry())
    (e,) = cfg.embedders
    assert e.column == "minilm_embedding"  # default output column
    assert cfg.input_specs == {"text": "text"}
    assert cfg.pipeline.on_empty_input == "skip"


def test_backend_kwargs_pass_through():
    e = EmbedderEntry.model_validate(entry(batch_size=128, dtype="bfloat16"))
    kwargs = e.backend_kwargs()
    assert kwargs == {"model": "org/model", "batch_size": 128, "dtype": "bfloat16"}


def test_modality_is_required():
    e = entry()
    del e["modality"]
    with pytest.raises(ValidationError, match="modality"):
        config(e)


def test_empty_embedders_list_rejected():
    with pytest.raises(ValidationError):
        config()


def test_duplicate_names_rejected():
    with pytest.raises(ValidationError, match="duplicate embedder names"):
        config(entry(), entry(output_column="other"))


def test_output_column_collision_rejected():
    with pytest.raises(ValidationError, match="output column collision"):
        config(
            entry(name="a", output_column="emb"),
            entry(name="b", output_column="emb"),
        )


def test_pooled_column_collides_too():
    with pytest.raises(ValidationError, match="output column collision"):
        config(
            entry(name="a", output_column="pooled"),
            entry(
                name="mv",
                kind="multivector",
                type="bge_m3",
                pooling={"type": "mean", "output_column": "pooled"},
            ),
        )


def test_conflicting_modalities_on_same_column_rejected():
    with pytest.raises(ValidationError, match="conflicting modalities"):
        config(
            entry(name="a"),
            entry(name="b", modality="image", output_column="b_emb"),
        )


def test_chunking_with_multiple_input_columns_rejected():
    with pytest.raises(ValidationError, match="more than one input_column"):
        config(
            entry(name="a", input_column="title"),
            entry(name="b", input_column="body"),
            chunking={"strategy": "fixed_char", "chunk_chars": 100},
        )


def test_chunking_with_multiple_inputs_ok_when_passthrough():
    cfg = config(
        entry(name="a", input_column="title"),
        entry(name="b", input_column="body"),
        chunking={"strategy": "passthrough"},
    )
    assert not cfg.chunking.splits


def test_chunking_with_non_text_modality_rejected():
    with pytest.raises(ValidationError, match="non-text modality"):
        config(
            entry(modality="image", input_column="img"),
            chunking={"strategy": "fixed_char", "chunk_chars": 100},
        )


def test_pooling_only_on_multivector():
    with pytest.raises(ValidationError, match="pooling"):
        config(entry(pooling={"type": "mean"}))


def test_pooling_default_column():
    cfg = config(
        entry(name="mv", kind="multivector", type="bge_m3", pooling={"type": "mean"})
    )
    assert cfg.embedders[0].pooled_column == "mv_pooled"


def test_max_length_only_for_text():
    with pytest.raises(ValidationError, match="max_length"):
        config(entry(modality="image", input_column="img", max_length=100))


def test_unknown_pipeline_key_rejected():
    with pytest.raises(ValidationError):
        config(entry(), pipeline={"max_text_length": 100})  # removed knob


def test_drop_columns_accepts_source_columns():
    cfg = config(entry(), pipeline={"drop_columns": ["image", "text"]})
    assert cfg.pipeline.drop_columns == ["image", "text"]


def test_drop_columns_rejects_embedding_outputs():
    with pytest.raises(ValidationError, match="drop_columns lists embedding output"):
        config(entry(), pipeline={"drop_columns": ["minilm_embedding"]})


def test_drop_columns_rejects_pooled_outputs():
    with pytest.raises(ValidationError, match="drop_columns lists embedding output"):
        config(
            entry(name="mv", kind="multivector", type="bge_m3", pooling={"type": "mean"}),
            pipeline={"drop_columns": ["mv_pooled"]},
        )
