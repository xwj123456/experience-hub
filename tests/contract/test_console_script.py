from importlib.metadata import entry_points

from typer import Typer
from typer.testing import CliRunner


def test_installed_console_script_loads_and_displays_help() -> None:
    matches = [
        entry_point
        for entry_point in entry_points(group="console_scripts")
        if entry_point.name == "experience-hub"
    ]

    assert len(matches) == 1
    assert matches[0].value == "experience_hub.cli.app:app"
    app = matches[0].load()
    assert isinstance(app, Typer)

    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Experience Hub" in result.output
