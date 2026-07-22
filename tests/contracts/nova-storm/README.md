# nova-storm conformance fixtures

Fixtures for `nova contract check` against `nova-storm-<backend>` executables.

`make test` runs, at `shape`/`dry-run` level (no live backend needed):

```bash
nova-contract check <nova-storm-qdrant> --contract contracts/nova-storm/v1.yaml
```

This validates the backend's `capabilities --json` against the canonical
contract in `contracts/nova-storm/v1.yaml`. The `--fixtures <dir>` flag points
here and is reserved for future fixture-driven dry-run checks; the checker
tolerates its absence.

Live conformance (actually storming a store) is exercised separately against a
real Qdrant — see the "Testing against a live Qdrant" section in `AGENTS.md`.
