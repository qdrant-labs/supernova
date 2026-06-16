import textwrap

from supernova.cli.skypilot_utils import resolve_resources


def _write(tmp_path, body, monkeypatch):
    f = tmp_path / "resources.yaml"
    f.write_text(textwrap.dedent(body))
    monkeypatch.setenv("NOVA_RESOURCES_FILE", str(f))


def test_absent_file_falls_back_to_builtin(tmp_path, monkeypatch):
    monkeypatch.setenv("NOVA_RESOURCES_FILE", str(tmp_path / "missing.yaml"))
    assert resolve_resources("load", None, None, {"cpus": 4}) == {"cpus": 4}


def test_layers_builtin_file_config_cli(tmp_path, monkeypatch):
    _write(
        tmp_path,
        """
        all:  {cloud: aws}
        load: {cpus: 8, use_spot: true}
        """,
        monkeypatch,
    )
    builtin = {"cpus": 4, "memory": 16, "cloud": "gcp", "use_spot": True}
    r = resolve_resources(
        "load",
        config_resources={"memory": 32},
        overrides={"cpus": 16, "cloud": None},  # None override is ignored
        builtin=builtin,
    )
    assert r["cloud"] == "aws"     # file `all` beats builtin
    assert r["use_spot"] is True   # file `load`
    assert r["memory"] == 32       # config beats file/builtin
    assert r["cpus"] == 16         # cli beats everything; None cloud didn't clobber


def test_tool_section_beats_all_section(tmp_path, monkeypatch):
    _write(tmp_path, "all: {cpus: 2}\nstorm: {cpus: 9}\n", monkeypatch)
    assert resolve_resources("storm", None, None, {})["cpus"] == 9
    assert resolve_resources("load", None, None, {})["cpus"] == 2  # only `all` applies


def test_cli_override_drops_none_keeps_false(tmp_path, monkeypatch):
    monkeypatch.setenv("NOVA_RESOURCES_FILE", str(tmp_path / "missing.yaml"))
    r = resolve_resources(
        "load", None, {"use_spot": False, "cloud": None}, {"use_spot": True, "cloud": "aws"}
    )
    assert r["use_spot"] is False  # explicit False overrides
    assert r["cloud"] == "aws"     # None override ignored, builtin kept
