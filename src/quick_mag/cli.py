"""Unified ``quick-mag`` command-line entry point.

Subcommands:
  * ``quick-mag build ...`` — build perovskite structures (with structural scans
    and element combinations) and write them as CIFs (see
    :mod:`quick_mag.build_cli`).
  * ``quick-mag solve STRUCTURE ...`` — run the oxidation-state / exchange / spin-config
    pipeline (see :mod:`quick_mag.magnetic_cli`).
  * ``quick-mag ui`` — launch the interactive Dear ImGui desktop application
    (requires the optional ``imgui-bundle`` dependency: ``pip install 'quick_mag[ui]'``).
"""

from __future__ import annotations

import argparse
import sys

from quick_mag.build_cli import (
    BUILD_DESCRIPTION,
    configure_build_parser,
    run_build,
)
from quick_mag.magnetic_cli import (
    SOLVE_DESCRIPTION,
    configure_solve_parser,
    run_solve,
)


def _launch_ui(_args) -> int:
    """Import and run the imgui UI, with a friendly message if the extra is missing."""
    try:
        from quick_mag import quick_mag_ui
    except ImportError as exc:
        print(
            "The interactive UI requires the 'imgui-bundle' dependency.\n"
            "Install it with:  pip install 'quick_mag[ui]'\n"
            f"(import error: {exc})",
            file=sys.stderr,
        )
        return 1
    quick_mag_ui.main()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quick-mag",
        description="Perovskite/crystal magnetism toolkit: structure builder, "
        "collinear-spin solver, and interactive visualization UI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser_ = subparsers.add_parser(
        "build", help="Build perovskite structures (scans, element combinations).",
        description=BUILD_DESCRIPTION,
    )
    configure_build_parser(build_parser_)
    build_parser_.set_defaults(func=run_build)

    solve_parser = subparsers.add_parser(
        "solve", help="Predict oxidation states, exchange, and spin configurations.",
        description=SOLVE_DESCRIPTION,
    )
    configure_solve_parser(solve_parser)
    solve_parser.set_defaults(func=run_solve)

    ui_parser = subparsers.add_parser(
        "ui", help="Launch the interactive desktop visualization UI.",
        description="Open the Dear ImGui builder/visualization window (requires "
        "'quick_mag[ui]').",
    )
    ui_parser.set_defaults(func=_launch_ui)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
