"""Prediction math + pass planning + CLI routing (no models, no network)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("obstore")  # engine import pulls in the storage package

import fake_backends  # noqa: F401 — registers fake / fake_fused backends

from nova_embed.config import EmbedConfig
from nova_embed.predict import (
    combine_texts_per_s,
    estimate_cost,
    plan_passes,
    predict_throughput,
    simulate_padding,
)


def config(entries: list[dict]) -> EmbedConfig:
    return EmbedConfig.model_validate(
        {
            "source": {"type": "huggingface", "dataset_name": "org/data"},
            "embedders": entries,
            "storage": {"type": "local", "output_dir": "/tmp/x"},
        }
    )


def entry(**overrides) -> dict:
    data = {
        "name": "d",
        "kind": "dense",
        "type": "fake",
        "model": "m",
        "input_column": "text",
        "modality": "text",
    }
    data.update(overrides)
    return data


# ------------------------------------------------------------- pass planning

def test_fused_entries_plan_one_pass():
    cfg = config(
        [
            entry(type="fake_fused", batch_size=128),
            entry(name="s", kind="sparse", type="fake_fused", batch_size=32,
                  output_column="s_col"),
        ]
    )
    passes, skipped = plan_passes(cfg)
    (p,) = passes
    assert p.fused and p.entries == ["d", "s"]
    assert p.batch_size == 32  # min rule, mirroring the engine
    assert skipped == []


def test_unfused_entries_plan_separate_passes():
    cfg = config([entry(), entry(name="e2", model="m2", output_column="c2")])
    passes, _ = plan_passes(cfg)
    assert [p.entries for p in passes] == [["d"], ["e2"]]
    assert all(not p.fused for p in passes)


def test_non_text_entries_are_skipped():
    cfg = config(
        [entry(), entry(name="img", type="fake_image", modality="image",
                        input_column="image_col", output_column="ic")]
    )
    passes, skipped = plan_passes(cfg)
    assert [p.entries for p in passes] == [["d"]]
    assert len(skipped) == 1 and "img" in skipped[0]


# ----------------------------------------------------------------- the model

def test_simulate_padding_uniform_lengths_is_perfectly_efficient():
    lengths = np.full(1000, 128)
    sim = simulate_padding(lengths, cutoff=512, batch_size=8, num_batches=100)
    assert sim["eta"] == pytest.approx(1.0)
    assert sim["pct_texts_truncated"] == 0.0
    assert sim["mean_truncated_tokens"] == 128


def test_simulate_padding_cutoff_truncates():
    lengths = np.array([100] * 90 + [10_000] * 10)
    sim = simulate_padding(lengths, cutoff=512, batch_size=8, num_batches=500)
    assert sim["pct_texts_truncated"] == pytest.approx(10.0)
    assert sim["mean_truncated_tokens"] == pytest.approx(0.9 * 100 + 0.1 * 512)
    assert 0 < sim["eta"] < 1


def test_predict_throughput_formula():
    # 100 TFLOPS, 1B params -> T_max = 1e14 / 2e9 = 50_000 tok/s
    out = predict_throughput(
        params=1_000_000_000, gpu_tflops=100, cutoff=512,
        eta=0.5, mean_tokens_per_text=100,
    )
    assert out["t_max_tok_s"] == pytest.approx(50_000)
    assert out["useful_tok_s"] == pytest.approx(25_000)
    assert out["texts_per_s"] == pytest.approx(250)


def test_combined_rate_is_harmonic():
    # two equal passes halve throughput; a fast pass barely moves a slow one
    assert combine_texts_per_s([100, 100]) == pytest.approx(50)
    assert combine_texts_per_s([1e9, 100]) == pytest.approx(100, rel=1e-3)


def test_estimate_cost():
    cost = estimate_cost(total_rows=3_600_000, texts_per_s=1000, rate_per_hr=2.0)
    assert cost["gpu_hours"] == pytest.approx(1.0)
    assert cost["raw_cost"] == pytest.approx(2.0)
    assert cost["total_cost"] == pytest.approx(2.4)


# ---------------------------------------------------------------- CLI routing

def test_cli_group_routes_default_and_subcommands(tmp_path):
    from click.testing import CliRunner

    from nova_embed.cli import cli

    runner = CliRunner()

    # bare config path (the original interface) falls through to `run`
    missing = tmp_path / "nope.yaml"
    res = runner.invoke(cli, [str(missing)])
    assert res.exit_code != 0
    assert "No such file" in str(res.output) + str(res.exception)

    # option-first also routes to run (pre-group behavior)
    res = runner.invoke(cli, ["--dry-run", str(missing)])
    assert "no such option" not in res.output.lower()

    # explicit subcommands resolve
    for args in (["run", "--help"], ["predict", "--help"]):
        res = runner.invoke(cli, args)
        assert res.exit_code == 0, res.output

    # group help lists both
    res = runner.invoke(cli, ["--help"])
    assert res.exit_code == 0
    assert "run" in res.output and "predict" in res.output
