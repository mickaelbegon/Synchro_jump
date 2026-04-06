"""Command-line entry point for Synchro Jump."""

from __future__ import annotations


def main() -> None:
    """Launch the interactive GUI."""

    from synchro_jump.gui.app import launch_app

    launch_app()


if __name__ == "__main__":
    main()
