from typer.testing import CliRunner
from cidre.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "cidre" in result.output


def test_status_no_config(tmp_path, monkeypatch):
    monkeypatch.setattr("cidre.cli.CIDRE_HOME", tmp_path / ".cidre")
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "not initialized" in result.output.lower() or "cidre init" in result.output.lower()
