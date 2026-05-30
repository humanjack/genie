"""Sanity: package imports and CLI stub runs."""

from genie import __version__
from genie.cli import main


def test_version_is_string():
    assert isinstance(__version__, str)
    assert __version__


def test_cli_help_exits_zero(capsys):
    rc = main(["--help"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "genie" in captured.out


def test_cli_unknown_exits_two(capsys):
    rc = main(["nope"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "unknown" in captured.err
