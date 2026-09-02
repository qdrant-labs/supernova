"""`nova_embed.manifest.code_versions` — provenance for WHICH CODE embedded.

Mirrors the nova-bf fix for the same bug: `git -C <package dir>` walks upward,
so a wheel install sitting inside some unrelated checkout gets a confident sha
for the wrong repository. A plausible-but-wrong commit is worse than none,
because `git_dirty` is scoped to the package directory and would read False.
"""
from __future__ import annotations

from nova_embed import manifest as m


def test_reports_git_for_an_in_repo_checkout():
    info = m.code_versions()
    assert info.get("git_commit"), "in-repo run should report a commit"
    assert len(info["git_commit"]) >= 7


def test_omits_git_when_the_repo_does_not_contain_this_package(monkeypatch, tmp_path):
    # Patch the module's own __file__ (which pkg_dir derives from). Patching
    # os.path.abspath would also change os.path.realpath, which calls it
    # internally — the check would then compare two fake paths and agree.
    pkg = tmp_path / "site-packages" / "nova_embed"
    pkg.mkdir(parents=True)
    monkeypatch.setattr(m, "__file__", str(pkg / "manifest.py"))

    class R:
        def __init__(self, out): self.stdout = out

    def fake_run(cmd, **kw):
        if "--show-toplevel" in cmd:
            return R("/some/other/repo\n")     # does NOT contain pkg_dir
        return R("deadbeef\n")

    monkeypatch.setattr(m.subprocess, "run", fake_run)
    info = m.code_versions()
    assert "git_commit" not in info, "git answered about an unrelated repo"
    assert "git_describe" not in info and "git_branch" not in info
    assert info.get("python_version"), "non-git provenance must survive"


def test_job_identity_reports_host_and_pid():
    info = m.job_identity()
    assert info.get("pid")
    assert "hostname" in info
