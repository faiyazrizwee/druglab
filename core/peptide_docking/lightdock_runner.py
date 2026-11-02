"""
Subprocess wrapper and execution manager for LightDock 0.9.4+.
"""

from __future__ import annotations

import os
import sys
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.peptide_docking.config import LightdockConfig
from utils.io import ensure_dir
from utils.logger import setup_logger

logger = setup_logger("lightdock_runner", level="INFO")


# Standard amino acids supported by LightDock potential functions (AMBER94 / DFIRE)
STANDARD_LIGHTDOCK_AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HID", "HIE", "HIP", "MSE", "CYX", "ASH", "GLH", "LYN"
}


def sanitize_structure_for_lightdock(input_pdb: str | Path, output_pdb: str | Path) -> None:
    """
    Sanitize PDB structure for LightDock simulation.
    Strips unsupported heteroatoms (metal ions, non-standard ligands, waters)
    to prevent LightDock scoring function failures (e.g. NotSupportedInScoringError in CPyDock/DFIRE).
    """
    input_path = Path(input_pdb)
    output_path = Path(output_pdb)
    lines_written = 0
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if line.startswith(("ATOM", "HETATM")):
                res_name = line[17:20].strip().upper()
                # Exclude water, ions, and non-standard hetero residues
                if res_name in {"HOH", "WAT", "H2O", "TIP3", "MG", "ZN", "CA", "FE", "MN", "CU", "NA", "K", "CL"}:
                    continue
                if res_name not in STANDARD_LIGHTDOCK_AMINO_ACIDS:
                    continue
                # Normalize HETATM to ATOM for standard residues
                clean_line = "ATOM  " + line[6:] if line.startswith("HETATM") else line
                f_out.write(clean_line)
                lines_written += 1
            elif line.startswith(("TER", "ENDMDL", "END")):
                f_out.write(line)

    if lines_written == 0:
        # Fallback if no lines matched standard residues
        shutil.copyfile(str(input_path), str(output_path))


@dataclass
class LightDockRunResult:
    candidate_id: str
    status: str  # "completed", "failed", "timeout", "missing_executable"
    exit_code: int = -1
    runtime_sec: float = 0.0
    work_dir: str = ""
    command_log: list[dict[str, Any]] = field(default_factory=list)
    setup_file: str | None = None
    rank_file: str | None = None
    poses: list[dict[str, Any]] = field(default_factory=list)
    error_message: str | None = None
    lightdock_version: str = ""


def find_lightdock_executable(name: str = "lightdock3.py", user_path: str | None = None) -> str:
    """
    Find a LightDock executable from user path, environment PATH, or current sys.prefix.
    """
    if user_path and os.path.isfile(user_path) and os.access(user_path, os.X_OK):
        return str(Path(user_path).resolve())

    # Try PATH
    exe = shutil.which(name)
    if exe:
        return exe

    # Try sys.prefix/bin
    prefix_bin = Path(sys.prefix) / "bin" / name
    if prefix_bin.is_file() and os.access(str(prefix_bin), os.X_OK):
        return str(prefix_bin.resolve())

    # Fallback name variants (e.g. without '3' or .py)
    variants = [
        name,
        name.replace("3", ""),
        name.replace(".py", ""),
        name.replace("3.py", ""),
    ]
    for var in variants:
        found = shutil.which(var)
        if found:
            return found
        p_var = Path(sys.prefix) / "bin" / var
        if p_var.is_file() and os.access(str(p_var), os.X_OK):
            return str(p_var.resolve())

    raise FileNotFoundError(
        f"LightDock executable '{name}' is not found in the environment.\n"
        f"Please install LightDock using: python utils/install_lightdock.py\n"
        f"or specify the executable path in the configuration."
    )


def get_lightdock_version() -> str:
    """Get installed LightDock version string."""
    try:
        setup_exe = find_lightdock_executable("lightdock3_setup.py")
        proc = subprocess.run([setup_exe, "-V"], capture_output=True, text=True, check=False)
        out = (proc.stdout + proc.stderr).strip()
        return out if out else "0.9.4"
    except Exception:
        return "unknown"


def run_lightdock_docking(
    candidate_id: str,
    receptor_pdb: str | Path,
    peptide_pdb: str | Path,
    output_dir: str | Path,
    config: LightdockConfig,
    restraints_path: str | Path | None = None,
    num_poses_to_generate: int = 5,
    cpu_cores: int = 4,
) -> LightDockRunResult:
    """
    Run the full LightDock simulation pipeline for a receptor–peptide pair:
    1. lightdock3_setup.py
    2. lightdock3.py
    3. lgd_rank.py
    4. lgd_generate_conformations.py (for top poses)
    """
    start_time = time.time()
    work_dir = ensure_dir(Path(output_dir) / f"dock_{candidate_id}")
    log_file = Path(work_dir) / "docking.log"
    command_log: list[dict[str, Any]] = []

    # Check executables
    try:
        setup_exe = find_lightdock_executable("lightdock3_setup.py")
        dock_exe = find_lightdock_executable("lightdock3.py", user_path=config.executable)
        rank_exe = find_lightdock_executable("lgd_rank.py")
        conf_exe = find_lightdock_executable("lgd_generate_conformations.py")
    except FileNotFoundError as e:
        return LightDockRunResult(
            candidate_id=candidate_id,
            status="missing_executable",
            error_message=str(e),
            work_dir=str(work_dir),
        )

    version_str = get_lightdock_version()
    swarms, steps, glowworms = config.resolve_sampling()

    # Copy & sanitize input structures into work_dir to keep LightDock setup contained
    local_rec = Path(work_dir) / "receptor.pdb"
    local_pep = Path(work_dir) / "peptide.pdb"
    sanitize_structure_for_lightdock(receptor_pdb, local_rec)
    sanitize_structure_for_lightdock(peptide_pdb, local_pep)

    local_rst = None
    if restraints_path and os.path.isfile(restraints_path):
        local_rst = Path(work_dir) / "restraints.list"
        shutil.copyfile(str(restraints_path), str(local_rst))

    def _execute(cmd: list[str], label: str, timeout: int | None = None) -> tuple[int, str, str]:
        cmd_start = time.time()
        logger.debug(f"[{candidate_id}] Running {label}: {' '.join(cmd)}")
        try:
            res = subprocess.run(
                cmd,
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            elapsed = time.time() - cmd_start
            command_log.append({
                "step": label,
                "cmd": cmd,
                "exit_code": res.returncode,
                "runtime_sec": round(elapsed, 2),
                "stdout_sample": res.stdout[-500:] if res.stdout else "",
                "stderr_sample": res.stderr[-500:] if res.stderr else "",
            })
            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(f"\n--- {label} (exit: {res.returncode}, time: {elapsed:.2f}s) ---\n")
                lf.write(f"Command: {' '.join(cmd)}\n")
                if res.stdout:
                    lf.write(f"STDOUT:\n{res.stdout}\n")
                if res.stderr:
                    lf.write(f"STDERR:\n{res.stderr}\n")
            return res.returncode, res.stdout, res.stderr
        except subprocess.TimeoutExpired as te:
            elapsed = time.time() - cmd_start
            msg = f"Command timed out after {timeout}s: {' '.join(cmd)}"
            logger.error(f"[{candidate_id}] {msg}")
            command_log.append({
                "step": label,
                "cmd": cmd,
                "exit_code": -9,
                "runtime_sec": round(elapsed, 2),
                "error": msg,
            })
            return -9, "", msg
        except Exception as ex:
            elapsed = time.time() - cmd_start
            msg = f"Execution error: {ex}"
            logger.error(f"[{candidate_id}] {msg}")
            return -1, "", msg

    # --- STEP 1: Setup ---
    setup_cmd = [
        setup_exe,
        "receptor.pdb",
        "peptide.pdb",
        "-s", str(swarms),
        "-g", str(glowworms),
        "--seed_points", str(config.seed),
        "--now",  # Remove H2O
        "--noxt", # Remove OXT
        "--noh",  # Remove Hydrogen atoms for scoring function compatibility
    ]
    if local_rst:
        setup_cmd.extend(["-r", "restraints.list"])
    if config.anm:
        setup_cmd.append("-anm")

    ret, stdout, stderr = _execute(setup_cmd, "lightdock_setup", timeout=300)
    if ret != 0:
        return LightDockRunResult(
            candidate_id=candidate_id,
            status="failed",
            exit_code=ret,
            runtime_sec=round(time.time() - start_time, 2),
            work_dir=str(work_dir),
            command_log=command_log,
            error_message=f"LightDock setup failed with code {ret}: {stderr or stdout}",
            lightdock_version=version_str,
        )

    setup_json = Path(work_dir) / "setup.json"
    if not setup_json.is_file():
        return LightDockRunResult(
            candidate_id=candidate_id,
            status="failed",
            work_dir=str(work_dir),
            command_log=command_log,
            error_message="LightDock setup completed but setup.json was not created.",
            lightdock_version=version_str,
        )

    # Determine actual swarms generated by setup
    actual_swarms = swarms
    try:
        import json
        with open(setup_json, "r", encoding="utf-8") as f:
            setup_data = json.load(f)
            if "swarms" in setup_data and isinstance(setup_data["swarms"], int):
                actual_swarms = setup_data["swarms"]
            elif "num_swarms" in setup_data and isinstance(setup_data["num_swarms"], int):
                actual_swarms = setup_data["num_swarms"]
    except Exception:
        pass

    # --- STEP 2: Simulation ---
    scoring_arg = config.scoring_function
    if config.scoring_function == "combined":
        scoring_conf_path = Path(work_dir) / "scoring.conf"
        weights = getattr(config, "scoring_weights", None) or {"cpydock": 0.5, "dfire2": 0.5}
        with open(scoring_conf_path, "w", encoding="utf-8") as scf:
            scf.write("# LightDock combined scoring configuration\n")
            for sf_name, weight in weights.items():
                scf.write(f"{sf_name} {weight}\n")
        scoring_arg = "scoring.conf"
    elif getattr(config, "custom_scoring_file", None) and os.path.isfile(config.custom_scoring_file):
        scoring_conf = Path(config.custom_scoring_file)
        dest_conf = Path(work_dir) / scoring_conf.name
        shutil.copyfile(str(scoring_conf), str(dest_conf))
        scoring_arg = scoring_conf.name
    elif os.path.isfile(config.scoring_function):
        scoring_conf = Path(config.scoring_function)
        dest_conf = Path(work_dir) / scoring_conf.name
        shutil.copyfile(str(scoring_conf), str(dest_conf))
        scoring_arg = scoring_conf.name

    sim_cmd = [
        dock_exe,
        "setup.json",
        str(steps),
        "-s", scoring_arg,
        "-c", str(cpu_cores),
        "-sg", str(config.seed),
    ]

    ret, stdout, stderr = _execute(sim_cmd, "lightdock_simulation", timeout=config.timeout_seconds)
    if ret == -9:
        return LightDockRunResult(
            candidate_id=candidate_id,
            status="timeout",
            exit_code=-9,
            runtime_sec=round(time.time() - start_time, 2),
            work_dir=str(work_dir),
            command_log=command_log,
            error_message=f"Simulation timed out after {config.timeout_seconds}s",
            lightdock_version=version_str,
        )
    if ret != 0 or "[lightdock] ERROR:" in (stdout or ""):
        err_msg = stderr or stdout
        for line in (stdout or "").splitlines():
            if "ERROR" in line:
                err_msg = line.strip()
                break
        return LightDockRunResult(
            candidate_id=candidate_id,
            status="failed",
            exit_code=ret if ret != 0 else 1,
            runtime_sec=round(time.time() - start_time, 2),
            work_dir=str(work_dir),
            command_log=command_log,
            error_message=f"LightDock simulation failed: {err_msg}",
            lightdock_version=version_str,
        )

    # --- STEP 3: Ranking ---
    rank_cmd = [
        rank_exe,
        str(actual_swarms),
        str(steps),
    ]
    ret, stdout, stderr = _execute(rank_cmd, "lightdock_rank", timeout=300)
    rank_file = Path(work_dir) / "rank_by_scoring.list"
    if not rank_file.is_file():
        # Check alternative rank filename
        alt = list(Path(work_dir).glob("rank*.list"))
        if alt:
            rank_file = alt[0]

    # --- STEP 4: Conformation Generation for top poses ---
    poses_info: list[dict[str, Any]] = []
    poses_dir = ensure_dir(Path(work_dir) / "poses")

    if rank_file and rank_file.is_file():
        # Parse top entries from rank file
        # Format: Swarm Glowworm Coordinates RecID LigID Luciferin Neigh VR RMSD PDB Clashes Scoring
        top_entries = []
        with open(rank_file, "r", encoding="utf-8") as rf:
            for line in rf:
                line = line.strip()
                if not line or line.startswith(("#", "Swarm")):
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        swarm_id = int(parts[0])
                        glow_id = int(parts[1])
                        score = float(parts[-1])
                        pdb_name = parts[-3] if "lightdock_" in parts[-3] else f"lightdock_{glow_id}.pdb"
                        top_entries.append((swarm_id, glow_id, score, pdb_name))
                    except ValueError:
                        continue
                if len(top_entries) >= num_poses_to_generate:
                    break

        # Generate conformations for top swarms
        swarms_to_gen = set(e[0] for e in top_entries)
        for s_id in swarms_to_gen:
            swarm_out = Path(work_dir) / f"swarm_{s_id}" / f"gso_{steps}.out"
            if swarm_out.is_file():
                # lgd_generate_conformations automatically looks for lightdock_receptor.pdb
                # when given receptor.pdb
                gen_cmd = [
                    conf_exe,
                    "receptor.pdb",
                    "peptide.pdb",
                    f"swarm_{s_id}/gso_{steps}.out",
                    str(glowworms),
                ]
                _execute(gen_cmd, f"generate_conformations_swarm_{s_id}", timeout=120)

        # Move generated PDBs to poses/ directory
        for rank_idx, (s_id, g_id, sc, pdb_name) in enumerate(top_entries, start=1):
            cand_pdb = Path(work_dir) / f"swarm_{s_id}" / pdb_name
            if cand_pdb.is_file():
                dest_pdb = poses_dir / f"pose_{rank_idx}_swarm{s_id}_glow{g_id}.pdb"
                shutil.copyfile(str(cand_pdb), str(dest_pdb))
                poses_info.append({
                    "rank": rank_idx,
                    "swarm_id": s_id,
                    "glowworm_id": g_id,
                    "score": sc,
                    "scoring_function": config.scoring_function,
                    "pdb_path": str(dest_pdb.resolve()),
                })

    elapsed_total = round(time.time() - start_time, 2)
    return LightDockRunResult(
        candidate_id=candidate_id,
        status="completed" if poses_info else ("completed_no_poses" if rank_file and rank_file.is_file() else "failed"),
        exit_code=0,
        runtime_sec=elapsed_total,
        work_dir=str(work_dir),
        command_log=command_log,
        setup_file=str(setup_json) if setup_json.is_file() else None,
        rank_file=str(rank_file) if rank_file and rank_file.is_file() else None,
        poses=poses_info,
        lightdock_version=version_str,
    )
