#!/usr/bin/env python3
"""
Unified CLI for vectorforge.

  vf embed <config>           # embed locally
  vf embed-dist <config>      # embed distributed via SkyPilot pool
  vf partition <config>       # run the embed pipeline with a no-op embedder
  vf partition-dist <config>  # distributed no-op partition via SkyPilot pool
  vf load <config>            # load into vector store
  vf load-dist <config>       # load distributed via SkyPilot
  vf analysis <config>        # analyze a (distributed) embedding run
"""

import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("usage: vf <command> [args...]\n")
        print("commands:")
        print("  embed           Embed a dataset locally")
        print("  embed-dist      Embed distributed via SkyPilot pool")
        print("  partition       Run pipeline with no-op embedder (validate sharding without GPU)")
        print("  partition-dist  Distributed partition via SkyPilot pool")
        print("  load            Load pre-embedded data into a vector store")
        print("  load-dist       Distribute loading across SkyPilot instances")
        print("  analysis        Analyze a (distributed) embedding run")
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
    elif command == "analysis":
        from cli.run_analysis import main as analysis_main
        analysis_main(argv)
    else:
        print(f"unknown command: {command}")
        print("run 'vf --help' for available commands")
        sys.exit(1)


if __name__ == "__main__":
    main()
