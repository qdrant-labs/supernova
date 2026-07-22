"""Engine assembly + run-time behavior, exercised through fake backends."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("obstore")  # nova_embed.embedders package pulls in storage

import fake_backends  # noqa: F401  — registers the fake (kind, "fake") backends
from fake_backends import DENSE_INSTANTIATIONS, FUSED_INSTANTIATIONS

from nova_embed.config import EmbedderEntry
from nova_embed.embedders.engine import build_engine
from nova_embed.models import MultiVectorEmbedding, OutputKind, SparseEmbedding


def entry(**overrides) -> EmbedderEntry:
    data = {
        "name": "dense_a",
        "kind": "dense",
        "type": "fake",
        "input_column": "text",
        "modality": "text",
    }
    data.update(overrides)
    return EmbedderEntry.model_validate(data)


def embed(engine, rows):
    return asyncio.run(engine.embed(rows))


def test_single_entry_roundtrip():
    engine = build_engine([entry()])
    out = embed(engine, [{"text": "abc"}, {"text": "hello"}])
    assert out == {"dense_a": [[3.0, 3.0], [5.0, 5.0]]}
    (spec,) = engine.output_specs
    assert spec.column == "dense_a_embedding"
    assert spec.kind == "dense"
    assert spec.model_name == "fake-dense"


def test_multiple_entries_same_column():
    engine = build_engine(
        [
            entry(name="a"),
            entry(name="b", kind="sparse", output_column="sparse_col"),
        ]
    )
    out = embed(engine, [{"text": "xy"}])
    assert out["a"] == [[2.0, 2.0]]
    assert out["b"] == [SparseEmbedding(indices=[2], values=[1.0])]


def test_entries_on_different_columns():
    engine = build_engine(
        [
            entry(name="title_emb", input_column="title"),
            entry(name="body_emb", input_column="body"),
        ]
    )
    out = embed(engine, [{"title": "ab", "body": "abcdef"}])
    assert out["title_emb"] == [[2.0, 2.0]]
    assert out["body_emb"] == [[6.0, 6.0]]


def test_empty_inputs_masked_to_none():
    engine = build_engine([entry()])
    out = embed(engine, [{"text": "abc"}, {"text": "   "}, {"text": None}, {"text": "z"}])
    assert out["dense_a"] == [[3.0, 3.0], None, None, [1.0, 1.0]]


def test_all_empty_batch_skips_model_call():
    engine = build_engine([entry()])
    out = embed(engine, [{"text": ""}, {"text": None}])
    assert out["dense_a"] == [None, None]


def test_max_length_truncates_per_entry():
    engine = build_engine(
        [entry(name="full"), entry(name="cut", max_length=3, output_column="cut_col")]
    )
    out = embed(engine, [{"text": "abcdefgh"}])
    assert out["full"] == [[8.0, 8.0]]
    assert out["cut"] == [[3.0, 3.0]]


def test_instance_sharing_same_backend_config():
    DENSE_INSTANTIATIONS.clear()
    engine = build_engine(
        [
            entry(name="a", input_column="title"),
            entry(name="b", input_column="body", output_column="b_col"),
        ]
    )
    assert len(DENSE_INSTANTIATIONS) == 1  # same kwargs -> one loaded model
    out = embed(engine, [{"title": "x", "body": "yy"}])
    assert out["a"] == [[1.0, 1.0]] and out["b"] == [[2.0, 2.0]]


def test_no_sharing_across_different_kwargs():
    DENSE_INSTANTIATIONS.clear()
    build_engine(
        [
            entry(name="a", model="m1"),
            entry(name="b", model="m2", output_column="b_col"),
        ]
    )
    assert len(DENSE_INSTANTIATIONS) == 2


def test_pooling_derives_dense_output():
    engine = build_engine(
        [
            entry(
                name="mv",
                kind="multivector",
                pooling={"type": "mean", "normalize": False},
            )
        ]
    )
    out = embed(engine, [{"text": "hi"}])
    assert isinstance(out["mv"][0], MultiVectorEmbedding)
    assert out["mv_pooled"] == [[1.5, 2.0]]  # mean of [3,0] and [0,4]
    columns = [s.column for s in engine.output_specs]
    assert columns == ["mv_embedding", "mv_pooled"]
    pooled_spec = engine.output_specs[1]
    assert pooled_spec.kind == "dense"
    assert "mean-pooled" in pooled_spec.model_name


def test_pooling_normalizes():
    engine = build_engine(
        [entry(name="mv", kind="multivector", pooling={"type": "max"})]
    )
    out = embed(engine, [{"text": "hi"}])
    vec = out["mv_pooled"][0]  # max -> [3,4], normalized -> [0.6, 0.8]
    assert vec == pytest.approx([0.6, 0.8])


def test_unsupported_modality_fails_before_instantiation():
    DENSE_INSTANTIATIONS.clear()
    with pytest.raises(ValueError, match="does not support modality 'image'"):
        build_engine([entry(modality="image", input_column="img")])
    assert DENSE_INSTANTIATIONS == []  # validated on the class, no weights loaded


def test_image_capable_backend_accepts_image_modality():
    engine = build_engine(
        [entry(type="fake_image", modality="image", input_column="img")]
    )
    assert engine.input_specs == {"img": "image"}


def test_unknown_type_mentions_other_kinds():
    with pytest.raises(ValueError, match="exists for kind"):
        build_engine([entry(kind="multivector", type="fake_image")])


# ------------------------------------------------------- multimodal entries

def mm_entry(**overrides) -> EmbedderEntry:
    data = {
        "name": "mm",
        "kind": "dense",
        "type": "fake_mm",
        "modality": "multimodal",
        "input_columns": {"text": "text", "image": "image"},
    }
    data.update(overrides)
    return EmbedderEntry.model_validate(data)


def test_multimodal_partial_rows_embed_present_parts():
    pytest.importorskip("PIL")
    from PIL import Image

    img = Image.new("RGB", (2, 2))
    engine = build_engine([mm_entry()])
    out = embed(
        engine,
        [
            {"text": "abc", "image": img},   # both parts
            {"text": "hello", "image": None},  # text-only: valid input
            {"text": "", "image": img},        # image-only: valid input
            {"text": None, "image": None},     # all parts empty: masked to None
        ],
    )
    # fake_mm encodes [len(text), has_image]
    assert out["mm"] == [[3.0, 1.0], [5.0, 0.0], [0.0, 1.0], None]


def test_multimodal_max_length_truncates_text_part():
    engine = build_engine([mm_entry(max_length=3)])
    out = embed(engine, [{"text": "abcdefg", "image": None}])
    assert out["mm"] == [[3.0, 0.0]]


def test_multimodal_gated_to_capable_backends():
    # the whole point of Modality.MULTIMODAL: a backend that never declared it
    # dies at build time, before any weights load
    with pytest.raises(ValueError, match="does not support modality 'multimodal'"):
        build_engine([mm_entry(type="fake")])


def test_multimodal_specs_groups_and_manifest_fields():
    engine = build_engine(
        [mm_entry(instruction="Represent the user's input."), entry(name="t")]
    )
    assert engine.input_specs == {"text": "text", "image": "image"}
    assert engine.input_groups == [
        {"text": "text", "image": "image"},
        {"text": "text"},
    ]
    mm_spec, t_spec = engine.output_specs
    assert mm_spec.input_column == "text=text,image=image"
    assert mm_spec.modality == "multimodal"
    assert mm_spec.instruction == "Represent the user's input."
    assert t_spec.instruction is None


# ------------------------------------------------------------------- fusion

def fused_entry(**overrides) -> EmbedderEntry:
    overrides.setdefault("model", "m")
    return entry(type="fake_fused", **overrides)


def test_fused_group_one_instance_one_pass():
    FUSED_INSTANTIATIONS.clear()
    DENSE_INSTANTIATIONS.clear()
    engine = build_engine(
        [
            fused_entry(name="d"),
            fused_entry(name="s", kind="sparse", output_column="s_col"),
            fused_entry(name="mv", kind="multivector", output_column="mv_col"),
        ]
    )
    # one fused instance covering all three kinds, no plain instantiation
    assert len(FUSED_INSTANTIATIONS) == 1
    assert FUSED_INSTANTIATIONS[0].kinds == {
        OutputKind.DENSE, OutputKind.SPARSE, OutputKind.MULTIVECTOR,
    }
    assert DENSE_INSTANTIATIONS == []

    out = embed(engine, [{"text": "abc"}, {"text": ""}])
    assert out["d"] == [[3.0, 3.0], None]  # empty inputs masked in fused path too
    assert out["s"] == [SparseEmbedding(indices=[3], values=[1.0]), None]
    assert isinstance(out["mv"][0], MultiVectorEmbedding)

    # one spec per entry — fusion is invisible downstream
    d_spec, s_spec, mv_spec = engine.output_specs
    assert [s.name for s in engine.output_specs] == ["d", "s", "mv"]
    assert d_spec.dimensions == 2 and s_spec.dimensions is None
    assert d_spec.kind == "dense" and mv_spec.kind == "multivector"


def test_fused_subset_of_kinds():
    FUSED_INSTANTIATIONS.clear()
    build_engine(
        [
            fused_entry(name="d"),
            fused_entry(name="s", kind="sparse", output_column="s_col"),
        ]
    )
    assert len(FUSED_INSTANTIATIONS) == 1
    assert FUSED_INSTANTIATIONS[0].kinds == {OutputKind.DENSE, OutputKind.SPARSE}


def test_no_fusion_across_different_inputs_or_kwargs():
    FUSED_INSTANTIATIONS.clear()
    DENSE_INSTANTIATIONS.clear()
    build_engine(
        [
            fused_entry(name="d"),
            fused_entry(name="s", kind="sparse", input_column="body",
                        output_column="s1"),  # different input column
            fused_entry(name="s2", kind="sparse", model="other",
                        output_column="s2_col"),  # different model
        ]
    )
    assert FUSED_INSTANTIATIONS == []  # three groups of one -> all plain units


def test_no_fusion_on_duplicate_kinds():
    FUSED_INSTANTIATIONS.clear()
    build_engine(
        [
            fused_entry(name="a"),
            fused_entry(name="b", output_column="b_col"),
            fused_entry(name="s", kind="sparse", output_column="s_col"),
        ]
    )
    assert FUSED_INSTANTIATIONS == []


def test_fused_batch_size_mismatch_warns_and_takes_min(caplog):
    FUSED_INSTANTIATIONS.clear()
    with caplog.at_level("WARNING"):
        build_engine(
            [
                fused_entry(name="d", batch_size=128),
                fused_entry(name="s", kind="sparse", batch_size=32,
                            output_column="s_col"),
            ]
        )
    assert len(FUSED_INSTANTIATIONS) == 1
    assert FUSED_INSTANTIATIONS[0].batch_size == 32
    assert any("batch_size" in r.message for r in caplog.records)


def test_fused_multivector_member_pools():
    engine = build_engine(
        [
            fused_entry(name="d"),
            fused_entry(name="mv", kind="multivector", output_column="mv_col",
                        pooling={"type": "mean", "normalize": False}),
        ]
    )
    out = embed(engine, [{"text": "hi"}])
    assert out["mv_pooled"] == [[1.5, 2.0]]  # mean of [3,0] and [0,4]
    pooled_spec = next(s for s in engine.output_specs if s.name == "mv_pooled")
    assert pooled_spec.kind == "dense"


def test_plain_type_never_fuses():
    FUSED_INSTANTIATIONS.clear()
    DENSE_INSTANTIATIONS.clear()
    build_engine(
        [
            entry(name="d", model="m"),
            entry(name="s", kind="sparse", model="m", output_column="s_col"),
        ]
    )
    assert FUSED_INSTANTIATIONS == []  # type "fake" has no fused registration
    assert len(DENSE_INSTANTIATIONS) == 1
