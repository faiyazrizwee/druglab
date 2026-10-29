"""
Unit tests for LightDock runner (executable discovery, mocked subprocess execution, timeouts).
"""

from unittest.mock import patch, MagicMock
from pathlib import Path
import subprocess

from core.peptide_docking.config import LightdockConfig
from core.peptide_docking.lightdock_runner import (
    find_lightdock_executable,
    get_lightdock_version,
    run_lightdock_docking,
)


def test_find_lightdock_executable():
    try:
        exe = find_lightdock_executable("lightdock3.py")
        assert exe is not None
        assert "lightdock3.py" in exe
    except FileNotFoundError:
        # Mock fallback when lightdock binary is not present in environment
        with patch("shutil.which", return_value="/mock/bin/lightdock3.py"):
            exe = find_lightdock_executable("lightdock3.py")
            assert exe == "/mock/bin/lightdock3.py"


def test_get_lightdock_version():
    ver = get_lightdock_version()
    if ver == "unknown":
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="LightDock 0.9.4\n", stderr="")
            with patch("shutil.which", return_value="/mock/bin/lightdock3.py"):
                ver = get_lightdock_version()
                assert "0.9.4" in ver or "lightdock" in ver.lower()
    else:
        assert "0.9.4" in ver or "lightdock" in ver.lower()


def test_run_lightdock_docking_mocked_success(tmp_path):
    rec_pdb = tmp_path / "rec.pdb"
    pep_pdb = tmp_path / "pep.pdb"
    rec_pdb.write_text("ATOM      1  N   ALA A   1       0.0   0.0   0.0  1.00  0.00           N\n")
    pep_pdb.write_text("ATOM      1  N   ALA B   1       1.0   1.0   1.0  1.00  0.00           N\n")

    cfg = LightdockConfig(preset="fast", swarms=1, steps=1, glowworms=10)

    # Mock subprocess.run
    def mock_subprocess_run(cmd, *args, **kwargs):
        cwd = Path(kwargs.get("cwd", tmp_path))
        # If setup is called, create setup.json
        if "lightdock3_setup.py" in cmd[0] or "setup" in cmd[0]:
            (cwd / "setup.json").write_text('{"swarms": 1, "glowworms": 10}')
            return MagicMock(returncode=0, stdout="setup done", stderr="")
        elif "lightdock3.py" in cmd[0]:
            return MagicMock(returncode=0, stdout="simulation done", stderr="")
        elif "lgd_rank.py" in cmd[0]:
            (cwd / "rank_by_scoring.list").write_text("0 0 -12.5 0.0\n")
            return MagicMock(returncode=0, stdout="ranking done", stderr="")
        elif "lgd_generate_conformations.py" in cmd[0]:
            s_dir = cwd / "swarm_0"
            s_dir.mkdir(exist_ok=True, parents=True)
            (s_dir / "lightdock_0.pdb").write_text("ATOM  1  N  ALA B 1  0.0 0.0 0.0\n")
            return MagicMock(returncode=0, stdout="conformations done", stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("core.peptide_docking.lightdock_runner.find_lightdock_executable", side_effect=lambda name, user_path=None: user_path or f"/mock/bin/{name}"):
        with patch("subprocess.run", side_effect=mock_subprocess_run):
            # We also need swarm_0/gso_1.out to trigger conformation generation
            # Let's run docking
            res = run_lightdock_docking(
                candidate_id="mock_cand",
                receptor_pdb=rec_pdb,
                peptide_pdb=pep_pdb,
                output_dir=tmp_path,
                config=cfg,
            )
            assert res.exit_code == 0
            assert res.candidate_id == "mock_cand"


def test_run_lightdock_docking_timeout(tmp_path):
    rec_pdb = tmp_path / "rec.pdb"
    pep_pdb = tmp_path / "pep.pdb"
    rec_pdb.write_text("ATOM      1  N   ALA A   1       0.0   0.0   0.0  1.00  0.00           N\n")
    pep_pdb.write_text("ATOM      1  N   ALA B   1       1.0   1.0   1.0  1.00  0.00           N\n")

    cfg = LightdockConfig(preset="fast", timeout_seconds=1)

    def mock_subprocess_timeout(cmd, *args, **kwargs):
        if "setup" in cmd[0]:
            cwd = Path(kwargs.get("cwd", tmp_path))
            (cwd / "setup.json").write_text('{"swarms": 1}')
            return MagicMock(returncode=0, stdout="", stderr="")
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=1)

    with patch("core.peptide_docking.lightdock_runner.find_lightdock_executable", side_effect=lambda name, user_path=None: user_path or f"/mock/bin/{name}"):
        with patch("subprocess.run", side_effect=mock_subprocess_timeout):
            res = run_lightdock_docking(
                candidate_id="timeout_cand",
                receptor_pdb=rec_pdb,
                peptide_pdb=pep_pdb,
                output_dir=tmp_path,
                config=cfg,
            )
            assert res.status == "timeout"
            assert res.exit_code == -9
