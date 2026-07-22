"""nova-embed CLI: a group whose default command is `run`.

Backward compatible with the original single-command interface: any first
argument that isn't a known subcommand routes to `run`, so
`nova embed <config> [flags]` keeps working exactly as before, while
`nova embed predict <config>` (and the explicit `nova embed run <config>`)
address subcommands. The only unreachable spelling is a config file literally
named like a subcommand with no path separator — write `./predict` if you
must.
"""

import click

from nova_embed.cli.predict import predict
from nova_embed.cli.run_embedder import embed


class DefaultCommandGroup(click.Group):
    """Route unknown first arguments (config paths, options) to `run`."""

    default_command = "run"

    def parse_args(self, ctx, args):
        if not args:
            # preserve the original no-args behavior (`run` falls back to
            # NOVA_CONFIG_PATH) instead of printing group help
            args = [self.default_command]
        elif args[0] not in self.commands and args[0] not in ("-h", "--help"):
            args = [self.default_command, *args]
        return super().parse_args(ctx, args)


@click.group(cls=DefaultCommandGroup, name="nova-embed")
def cli():
    """Embed datasets (`run`, the default command) and plan runs (`predict`)."""


cli.add_command(embed)  # registered as "run" via its own decorator
cli.add_command(predict)
