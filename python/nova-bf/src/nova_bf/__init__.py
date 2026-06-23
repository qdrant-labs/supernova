"""Brute-force exact nearest-neighbor ground truth.

Two phases, both map-reduce:

  compute  — each worker iterates its slice of corpus files, holding a running
             top-K per query in GPU memory (intra-worker reduce), then writes a
             partial result. `nova dist bf compute --num-jobs N` runs many.
  merge    — combine the per-worker partials into one top-K per query
             (inter-worker reduce). Single GPU run (`--num-jobs 1`) needs no merge.

Hit ids are `make_point_id(filename, row)` — identical to the loader's
`vf_point_id` DuckDB macro — so brute-force hits line up with Qdrant point ids.
"""
