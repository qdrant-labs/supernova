"""Reusable three-way parity harness for nova-bf ground truth.

Every search nova-bf can express is checked against TWO independent oracles:

  * `naive` — a plain-Python reference (loops, `float`, no numpy vectorization,
    no shared code with `nova_bf`) that re-states the documented semantics of
    each metric, each vector_type and each filter condition. It is the oracle
    for *semantics*: if nova-bf and the naive reference disagree, one of them
    has the meaning of the search wrong.
  * `qdrant_ref` — a live Qdrant collection queried with `exact=True`, over the
    same points, with the nova-bf `Filter` translated into Qdrant's own filter
    language. It is the oracle for *fidelity to the engine the ground truth is
    scored against*: nova-bf exists to grade Qdrant's recall, so a GT that
    Qdrant's own exact search disagrees with is worthless no matter how
    internally consistent it is.

The three modalities (dense / sparse / multivector) and the whole filter
language are covered from one shared synthetic dataset (`corpus.py`), loaded
once into one Qdrant collection carrying every modality as a named vector, so
a filter × modality cross-product costs one upsert rather than one per cell.

Device reuse
------------
The harness never hardcodes a device. `runner.run` sets `NOVA_BF_DEVICE` for
the duration of a run, and `devices.parity_devices()` reports which devices
this machine can answer on — `["cpu"]` on a laptop, `["cpu", "cuda"]` on a GPU
box. The same test file therefore does strictly more work on a GPU machine
without being edited: every case runs on both devices, and
`test_parity_devices.py` additionally asserts the two devices agree with each
other. That is the "eventually do the same on a GPU" path — there is nothing
to port, only a machine with a GPU to run it on.

See `tests/parity/README.md` for how to run it.
"""
