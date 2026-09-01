"""Unified ``quick-mag`` command-line entry point.

Subcommands:
  * ``quick-mag build ...`` — build perovskite structures (with structural scans
    and element combinations) and write them as CIFs (see
    :mod:`quick_mag.build_cli`).
  * ``quick-mag chgnet STRUCTURE ...`` — CHGNet single-point energies and geometry
    optimizations (see :mod:`quick_mag.chgnet_cli`; requires the optional
    ``chgnet`` extra: ``pip install -e '.[chgnet]'``).
  * ``quick-mag solve STRUCTURE ...`` — run the oxidation-state / exchange / spin-config
    pipeline (see :mod:`quick_mag.magnetic_cli`).
  * ``quick-mag serve`` — run the calculation server a remote UI submits jobs to
    (see :mod:`quick_mag.remote.server`).
  * ``quick-mag ui`` — launch the interactive Dear ImGui desktop application
    (requires the optional ``imgui-bundle`` dependency: ``pip install -e '.[ui]'``).

Commands chain with ``::``, passing structures from one stage to the next in
memory instead of through files::

    quick-mag build --a-site La --b-site Mn :: chgnet :: solve

Only the last stage writes to disk, unless a stage is given an explicit ``-o``.
``solve`` must be last (its spin data feeds nothing else) and ``build`` must be
first (it consumes no structures).
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from quick_mag.build_cli import (
    BUILD_DESCRIPTION,
    build_structures,
    configure_build_parser,
    report_build,
)
from quick_mag.magnetic_cli import (
    SOLVE_DESCRIPTION,
    configure_solve_parser,
    run_solve,
)

# The token that separates chained stages. An ordinary shell word, so it needs no
# quoting (unlike '|' or '|>', which the shell would try to interpret).
CHAIN_TOKEN = "::"

# stage -> the stages allowed to follow it. ``solve`` is terminal because nothing
# consumes spin configurations, and ``ui`` is interactive so it never chains.
_ALLOWED_SUCCESSORS = {
    "build": {"chgnet", "solve"},
    "chgnet": {"chgnet", "solve"},
    "solve": set(),
    "ui": set(),
    "serve": set(),
}

# Stages that are long-running and interactive rather than structure transforms.
# They consume nothing and produce nothing, so they can never appear in a chain.
_STANDALONE_STAGES = ("ui", "serve")


def _launch_ui(_args, **_kwargs) -> int:
    """Import and run the imgui UI, with a friendly message if the extra is missing."""
    try:
        from quick_mag import quick_mag_ui
    except ImportError as exc:
        print(
            "The interactive UI requires the 'imgui-bundle' dependency.\n"
            "Install it from the repository root with:  pip install -e '.[ui]'\n"
            f"(import error: {exc})",
            file=sys.stderr,
        )
        return 1
    quick_mag_ui.main()
    return 0


def _run_serve(args) -> int:
    """Start the remote-calculation server, or explain what is missing.

    Imported lazily for the same reason as the chgnet command: the Pyodide build
    ships neither the server module nor anything it would need.
    """
    try:
        from quick_mag.remote.server import run_serve
    except ImportError as exc:
        print(f"The 'serve' command is unavailable (import error: {exc})", file=sys.stderr)
        return 1
    return run_serve(args)


def _configure_serve_parser(parser: argparse.ArgumentParser) -> None:
    try:
        from quick_mag.remote.server import SERVE_DESCRIPTION, configure_serve_parser
    except ImportError:
        parser.add_argument("serve_args", nargs=argparse.REMAINDER)
        parser.set_defaults(serve_unavailable=True)
        return
    parser.description = SERVE_DESCRIPTION
    configure_serve_parser(parser)


def _configure_chgnet_parser(parser: argparse.ArgumentParser) -> None:
    """Attach the ``chgnet`` arguments, tolerating a stripped-down install.

    The import is deferred so ``quick_mag.cli`` stays importable in the
    numpy/scipy-only Pyodide build, which does not ship the CHGNet modules at
    all; the subcommand then reports the install hint when it is used. CHGNet and
    ASE themselves are only imported once a calculation actually starts.
    """
    try:
        from quick_mag.chgnet_cli import CHGNET_DESCRIPTION, configure_chgnet_parser
    except ImportError:
        parser.add_argument("chgnet_args", nargs=argparse.REMAINDER)
        parser.set_defaults(chgnet_unavailable=True)
        return
    parser.description = CHGNET_DESCRIPTION
    configure_chgnet_parser(parser)


def _run_chgnet(args, structures=None, *, write: bool = True):
    """Dispatch to :mod:`quick_mag.chgnet_cli`, or explain how to install it."""
    if getattr(args, "chgnet_unavailable", False):
        raise ImportError(
            "The 'chgnet' command is not available in this installation.\n"
            "Install it from the repository root with:  pip install -e '.[chgnet]'"
        )
    from quick_mag.chgnet_cli import run_chgnet

    return run_chgnet(args, structures, write=write)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quick-mag",
        description="Perovskite/crystal magnetism toolkit: structure builder, "
        "CHGNet relaxations, collinear-spin solver, and interactive "
        "visualization UI. Chain commands with '::'.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser_ = subparsers.add_parser(
        "build", help="Build perovskite structures (scans, element combinations).",
        description=BUILD_DESCRIPTION,
    )
    configure_build_parser(build_parser_)
    build_parser_.set_defaults(stage="build")

    chgnet_parser = subparsers.add_parser(
        "chgnet", help="CHGNet single-point energies and geometry optimizations.",
    )
    _configure_chgnet_parser(chgnet_parser)
    chgnet_parser.set_defaults(stage="chgnet")

    solve_parser = subparsers.add_parser(
        "solve", help="Predict oxidation states, exchange, and spin configurations.",
        description=SOLVE_DESCRIPTION,
    )
    configure_solve_parser(solve_parser)
    solve_parser.set_defaults(stage="solve")

    serve_parser = subparsers.add_parser(
        "serve", help="Run the calculation server a remote quick-mag UI submits to.",
    )
    _configure_serve_parser(serve_parser)
    serve_parser.set_defaults(stage="serve")

    ui_parser = subparsers.add_parser(
        "ui", help="Launch the interactive desktop visualization UI.",
        description="Open the Dear ImGui builder/visualization window (requires the "
        "'ui' extra: pip install -e '.[ui]').",
    )
    ui_parser.set_defaults(stage="ui")

    return parser


def split_chain(argv: List[str]) -> List[List[str]]:
    """Split an argument list on bare ``::`` tokens into per-stage segments.

    Empty segments (a leading, trailing, or doubled ``::``) raise ``ValueError``
    rather than being silently dropped, since they always mean a typo.
    """
    segments: List[List[str]] = [[]]
    for token in argv:
        if token == CHAIN_TOKEN:
            segments.append([])
        else:
            segments[-1].append(token)
    if any(not segment for segment in segments) and len(segments) > 1:
        raise ValueError(
            f"Empty stage in the '{CHAIN_TOKEN}' chain: every '{CHAIN_TOKEN}' must "
            "sit between two commands."
        )
    return segments


def validate_chain(stages: List[argparse.Namespace]) -> None:
    """Check that the sequence of stages is one the pipeline can actually run."""
    if len(stages) == 1:
        return

    for index, args in enumerate(stages):
        name = args.stage
        first, last = index == 0, index == len(stages) - 1

        if name in _STANDALONE_STAGES:
            raise ValueError(
                f"'{name}' runs on its own and cannot be part of a "
                f"'{CHAIN_TOKEN}' chain."
            )
        if name == "build" and not first:
            raise ValueError(
                "'build' generates its own structures, so it can only be the "
                f"first stage of a '{CHAIN_TOKEN}' chain."
            )
        if not last and stages[index + 1].stage not in _ALLOWED_SUCCESSORS[name]:
            successor = stages[index + 1].stage
            if not _ALLOWED_SUCCESSORS[name]:
                raise ValueError(
                    f"'{name}' must be the last stage of a '{CHAIN_TOKEN}' chain; "
                    f"nothing can consume its output, so '{successor}' cannot follow it."
                )
            raise ValueError(f"'{successor}' cannot follow '{name}'.")
        if not first and getattr(args, "structures", None):
            raise ValueError(
                f"'{name}' receives its structures from the previous stage, so it "
                "cannot also be given structure files."
            )


def run_chain(stages: List[argparse.Namespace]) -> int:
    """Run each stage in turn, passing structures forward in memory.

    A stage writes to disk only when it is the last one, or when it was given an
    explicit ``-o/--output-dir``.
    """
    structures = None
    for index, args in enumerate(stages):
        last = index == len(stages) - 1
        if args.stage == "build":
            structures = build_structures(args)
            report_build(args, structures, write=last)
        elif args.stage == "chgnet":
            structures = _run_chgnet(args, structures, write=last)
        elif args.stage == "solve":
            return run_solve(args, structures)
        elif args.stage == "serve":
            return _run_serve(args)
        else:
            return _launch_ui(args)
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    try:
        segments = split_chain(argv)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    stages = [parser.parse_args(segment) for segment in segments]
    try:
        validate_chain(stages)
        return run_chain(stages)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
