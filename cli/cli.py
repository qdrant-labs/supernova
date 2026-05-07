#!/usr/bin/env python3
"""
Unified CLI for vectorforge.

  vf embed <config>                     # embed locally
  vf embed-dist <config>                # embed distributed via SkyPilot pool
  vf partition <config>                 # run the embed pipeline with a no-op embedder
  vf partition-dist <config>            # distributed no-op partition via SkyPilot pool
  vf load <config>                      # load into vector store
  vf load-dist <config>                 # load distributed via SkyPilot
  vf push-hf <s3> <repo>                # upload S3 parquets to HuggingFace Hub
  vf push-hf-dist <s3> <repo>           # distributed HF upload via SkyPilot pool
  vf generate-queries <s3> -n N         # sample N rows as eval queries (runs on EC2)
  vf brute-force <s3> --queries         # exhaustive nearest-neighbor search (single GPU EC2)
  vf brute-force-dist <s3> --queries    # distributed brute-force via GPU pool
  vf brute-force-merge <s3> --queries   # merge partial results from dist run
  vf analysis <config>                  # analyze a (distributed) embedding run
  vf throughput-predict <config>        # predict embedding throughput + cost from a config
"""

import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("usage: vf <command> [args...]\n")
        print("commands:")
        print("  embed              Embed a dataset locally")
        print("  embed-dist         Embed distributed via SkyPilot pool")
        print("  partition          Run pipeline with no-op embedder (validate sharding without GPU)")
        print("  partition-dist     Distributed partition via SkyPilot pool")
        print("  load               Load pre-embedded data into a vector store")
        print("  load-dist          Distribute loading across SkyPilot instances")
        print("  push-hf            Upload S3 parquets to a HuggingFace Hub dataset")
        print("  push-hf-dist       Distribute HF upload across SkyPilot instances")
        print("  generate-queries   Sample N rows as eval queries (launches EC2, --local to run here)")
        print("  brute-force        Exhaustive nearest-neighbor search for recall eval (single GPU)")
        print("  brute-force-dist   Distributed brute-force via SkyPilot GPU pool")
        print("  brute-force-merge  Merge partial results from a distributed run")
        print("  analysis           Analyze a (distributed) embedding run")
        print("  throughput-predict Predict embedding throughput + cost from a config")
        sys.exit(0)

    command = sys.argv[1]
    argv = sys.argv[2:]

    if command == "embed":
        from cli.run_embedder import main as embed_main
        embed_main(argv)
    elif command == "embed-dist":
        from cli.run_embed_distributed import main as embed_dist_main
        embed_dist_main(argv)
    elif command == "partition":
        from cli.run_partition import main as partition_main
        partition_main(argv)
    elif command == "partition-dist":
        from cli.run_partition_distributed import main as partition_dist_main
        partition_dist_main(argv)
    elif command == "load":
        from cli.run_loader import main as load_main
        load_main(argv)
    elif command == "load-dist":
        from cli.run_load_distributed import main as load_dist_main
        load_dist_main(argv)
    elif command == "push-hf":
        from cli.run_push_hf import main as push_hf_main
        push_hf_main(argv)
    elif command == "push-hf-dist":
        from cli.run_push_hf_distributed import main as push_hf_dist_main
        push_hf_dist_main(argv)
    elif command == "generate-queries":
        from cli.run_generate_queries import main as gen_queries_main
        gen_queries_main(argv)
    elif command == "brute-force":
        from cli.run_brute_force import main as brute_force_main
        brute_force_main(argv)
    elif command == "brute-force-dist":
        from cli.run_brute_force_distributed import main as brute_force_dist_main
        brute_force_dist_main(argv)
    elif command == "brute-force-merge":
        from cli.run_brute_force import main as brute_force_main
        brute_force_main(["--merge", *argv])
    elif command == "analysis":
        from cli.run_analysis import main as analysis_main
        analysis_main(argv)
    elif command == "throughput-predict":
        from cli.run_throughput_predict import main as throughput_predict_main
        throughput_predict_main(argv)
    else:
        print(f"unknown command: {command}")
        print("run 'vf --help' for available commands")
        sys.exit(1)


if __name__ == "__main__":
    main()
