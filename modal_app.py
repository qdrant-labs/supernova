"""
Modal entrypoint for vectorforge pipelines.

Usage:
  # Run a single config
  modal run modal_app.py --config configs/mteb_tweets_openai.yaml

  # Run with GPU
  modal run modal_app.py --config configs/local_gte.yaml --gpu

  # Run a batch file (each job gets its own compute settings)
  modal run modal_app.py --batch configs/batch.yaml

  # Run all individual configs in parallel (cpu)
  modal run modal_app.py
"""

import modal

app = modal.App("vectorforge")

DEPS = [
    "datasets",
    "pyarrow",
    "openai",
    "tiktoken",
    "tqdm",
    "aiobotocore",
    "huggingface_hub",
    "pyyaml",
    "httpx",
    "sentence-transformers",
    "torch",
]

LOCAL_DIRS = [
    ("vectorforge", "/app/vectorforge"),
    ("scripts", "/app/scripts"),
    ("configs", "/app/configs"),
]

image = modal.Image.debian_slim(python_version="3.11").pip_install(*DEPS)
for src, dst in LOCAL_DIRS:
    image = image.copy_local_dir(src, dst)

gpu_image = modal.Image.from_registry(
    "nvidia/cuda:12.1.0-runtime-ubuntu22.04", add_python="3.11"
).pip_install(*DEPS)
for src, dst in LOCAL_DIRS:
    gpu_image = gpu_image.copy_local_dir(src, dst)


def _run_pipeline(config_path: str):
    """Shared pipeline execution logic."""
    import sys
    sys.path.insert(0, "/app")
    import asyncio
    import yaml
    from scripts.run_pipeline import build_source, build_embedder, build_storage
    from vectorforge.pipeline.runner import run

    with open(f"/app/{config_path}") as f:
        config = yaml.safe_load(f)

    source = build_source(dict(config["source"]))
    embedder = build_embedder(dict(config["embedder"]))
    storage = build_storage(dict(config["storage"]))
    pipeline_cfg = config.get("pipeline", {})
    storage_cfg = config.get("storage", {})

    asyncio.run(
        run(
            source=source,
            embedder=embedder,
            storage=storage,
            chunk_size=pipeline_cfg.get("chunk_size", 10_000),
            max_tokens=pipeline_cfg.get("max_tokens", 8192),
            num_workers=pipeline_cfg.get("num_workers", 8),
            flush_threshold=pipeline_cfg.get("flush_threshold", 100_000),
            output_dir=storage_cfg.get("output_dir", "/tmp/vectorforge"),
        )
    )
    return f"Done: {config_path}"


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("vectorforge-secrets")],
    timeout=7200,
)
def run_cpu(config_path: str):
    return _run_pipeline(config_path)


@app.function(
    image=gpu_image,
    gpu="A10G",
    secrets=[modal.Secret.from_name("vectorforge-secrets")],
    timeout=7200,
)
def run_gpu(config_path: str):
    return _run_pipeline(config_path)


@app.local_entrypoint()
def main(
    config: str = "",
    batch: str = "",
    gpu: bool = False,
):
    import glob
    import yaml

    if batch:
        # Read batch file, dispatch each job with its own compute settings
        with open(batch) as f:
            batch_cfg = yaml.safe_load(f)

        jobs = batch_cfg["jobs"]
        print(f"Submitting {len(jobs)} jobs from {batch}")

        cpu_jobs = []
        gpu_jobs = []

        for job in jobs:
            cfg = job["config"]
            needs_gpu = job.get("gpu", False)
            print(f"  - {cfg} (gpu={needs_gpu})")

            if needs_gpu:
                gpu_jobs.append(cfg)
            else:
                cpu_jobs.append(cfg)

        # Submit CPU and GPU jobs in parallel
        cpu_handles = [run_cpu.spawn(cfg) for cfg in cpu_jobs]
        gpu_handles = [run_gpu.spawn(cfg) for cfg in gpu_jobs]

        for handle in cpu_handles + gpu_handles:
            print(handle.get())

    elif config:
        fn = run_gpu if gpu else run_cpu
        print(fn.remote(config))

    else:
        configs = sorted(glob.glob("configs/*.yaml"))
        # Exclude batch files
        configs = [c for c in configs if "batch" not in c]

        fn = run_gpu if gpu else run_cpu
        print(f"Submitting {len(configs)} jobs (gpu={gpu})")
        for cfg in configs:
            print(f"  - {cfg}")
        for result in fn.map(configs):
            print(result)