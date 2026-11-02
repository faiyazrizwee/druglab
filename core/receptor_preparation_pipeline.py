"""
DrugLab Unified Transparent Receptor Preparation Pipeline.

Executes user-confirmed structure preparation decisions for both Small-Molecule
(AutoDock Vina) and Peptide (LightDock) docking workflows:
1. Preserves original input structure file unchanged.
2. Generates separate prepared receptor structure files (prepared_receptor.pdb, protein.pdbqt).
3. Produces comprehensive, viewable/downloadable preparation reports in HTML, JSON, and TSV formats.
4. Enforces workflow safeguards and maintains a transparent audit trail of every component modified or removed.
"""

from __future__ import annotations

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Any, Literal

import gemmi
from utils.logger import setup_logger, log_step
from utils.io import ensure_dir, write_json
from core.structure_inspection import (
    KNOWN_METAL_IONS,
    KNOWN_COFACTORS,
    STANDARD_AMINO_ACIDS,
    NONSTANDARD_AA_MAPPINGS,
)

logger = setup_logger("receptor_preparation_pipeline", level="INFO")


def generate_html_preparation_report(
    report_data: dict[str, Any],
    output_path: str | Path,
) -> str:
    """Generate a clean, standalone HTML preparation audit report."""
    title = f"DrugLab Receptor Preparation Report — {report_data.get('receptor_name')}"
    workflow_mode = report_data.get("workflow_mode", "small_molecule")
    workflow_label = (
        "Protein–Small Molecule Docking (Vina)"
        if workflow_mode == "small_molecule"
        else "Protein–Peptide Docking (LightDock)"
    )

    rows_html = ""
    for item in report_data.get("decisions_log", []):
        act_class = "action-remove" if "remove" in item["action"] else "action-keep"
        rows_html += f"""
        <tr>
          <td><strong>{item['component']}</strong></td>
          <td><span class="badge {act_class}">{item['action'].upper()}</span></td>
          <td>{item['scientific_rationale']}</td>
          <td>{item['docking_consequences']}</td>
        </tr>
        """

    af_html = ""
    if report_data.get("is_alphafold"):
        af_stats = report_data.get("alphafold_caveats", {}).get("plddt_stats", {})
        af_html = f"""
        <div class="alert-box">
          <div style="font-weight:700;font-size:1.05rem;margin-bottom:6px;">Predicted Structure Notice (AlphaFold DB)</div>
          <p>{report_data.get("alphafold_caveats", {}).get("disclaimer", "")}</p>
          <div style="margin-top:10px;display:flex;gap:12px;font-weight:600;font-size:0.85rem;">
            <span>Mean pLDDT: {report_data.get("alphafold_caveats", {}).get("mean_plddt", "N/A")}</span>
            <span>Very High (>90): {af_stats.get("very_high_pct", 0)}%</span>
            <span>Confident (70-90): {af_stats.get("confident_pct", 0)}%</span>
            <span>Low (<70): {af_stats.get("low_pct", 0)}%</span>
          </div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #0f172a;
      color: #f1f5f9;
      line-height: 1.6;
      margin: 0;
      padding: 30px;
    }}
    .container {{
      max-width: 1000px;
      margin: 0 auto;
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 12px;
      padding: 30px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }}
    h1 {{
      font-size: 1.6rem;
      margin-top: 0;
      color: #38bdf8;
      border-bottom: 1px solid #334155;
      padding-bottom: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .subhead {{ font-size: 0.95rem; color: #94a3b8; font-weight: normal; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin: 20px 0;
    }}
    .metric-card {{
      background: #0f172a;
      border: 1px solid #334155;
      border-radius: 8px;
      padding: 16px;
    }}
    .metric-label {{ font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }}
    .metric-value {{ font-size: 1.25rem; font-weight: 700; color: #f8fafc; margin-top: 4px; }}
    .alert-box {{
      background: rgba(245, 158, 11, 0.12);
      border: 1px solid rgba(245, 158, 11, 0.35);
      border-radius: 8px;
      padding: 16px;
      margin: 20px 0;
      color: #fde68a;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 20px 0;
      font-size: 0.9rem;
    }}
    th, td {{
      padding: 12px 14px;
      text-align: left;
      border-bottom: 1px solid #334155;
    }}
    th {{
      background: #0f172a;
      color: #94a3b8;
      text-transform: uppercase;
      font-size: 0.78rem;
      letter-spacing: 0.05em;
    }}
    .badge {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
    }}
    .action-remove {{ background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }}
    .action-keep {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }}
    .footer {{
      margin-top: 30px;
      padding-top: 15px;
      border-top: 1px solid #334155;
      font-size: 0.82rem;
      color: #64748b;
      display: flex;
      justify-content: space-between;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>
      <span>DrugLab Receptor Preparation Report</span>
      <span class="subhead">{workflow_label}</span>
    </h1>

    <div class="grid">
      <div class="metric-card">
        <div class="metric-label">Target Receptor</div>
        <div class="metric-value">{report_data.get('receptor_name')}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Structure Source</div>
        <div class="metric-value">{report_data.get('structure_source')}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Residues (Before &rarr; After)</div>
        <div class="metric-value">{report_data.get('original_residues')} &rarr; {report_data.get('prepared_residues')}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Atoms (Before &rarr; After)</div>
        <div class="metric-value">{report_data.get('original_atoms')} &rarr; {report_data.get('prepared_atoms')}</div>
      </div>
    </div>

    {af_html}

    <h2 style="font-size:1.15rem; color:#e2e8f0; margin-top:25px;">Component Preparation Decisions & Audit Trail</h2>
    <table>
      <thead>
        <tr>
          <th style="width:25%;">Component</th>
          <th style="width:15%;">Action</th>
          <th style="width:35%;">Scientific Rationale</th>
          <th style="width:25%;">Docking Impact</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>

    <div class="grid" style="margin-top:20px;">
      <div class="metric-card">
        <div class="metric-label">Original Structure (Unmodified)</div>
        <div style="font-size:0.85rem; font-family:monospace; margin-top:4px; word-break:break-all;">{report_data.get('original_path')}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Prepared Receptor PDB</div>
        <div style="font-size:0.85rem; font-family:monospace; margin-top:4px; word-break:break-all;">{report_data.get('prepared_path')}</div>
      </div>
      {f'''<div class="metric-card">
        <div class="metric-label">Docking PDBQT File</div>
        <div style="font-size:0.85rem; font-family:monospace; margin-top:4px; word-break:break-all;">{report_data.get('pdbqt_path')}</div>
      </div>''' if report_data.get('pdbqt_path') else ''}
    </div>

    <div class="footer">
      <span>Generated by DrugLab 2.0</span>
      <span>{report_data.get('timestamp')}</span>
    </div>
  </div>
</body>
</html>
"""
    p = Path(output_path).resolve()
    with open(p, "w", encoding="utf-8") as f:
        f.write(html)
    return str(p)


def prepare_receptor_transparently(
    input_pdb: str | Path,
    output_dir: str | Path,
    workflow_mode: Literal["small_molecule", "peptide"] = "small_molecule",
    decisions: dict[str, Any] | None = None,
    inspection_data: dict[str, Any] | None = None,
    ph: float = 7.4,
) -> dict[str, Any]:
    """
    Executes transparent preparation of a receptor PDB structure based on user-reviewed decisions.

    - Preserves input PDB unchanged.
    - Exports prepared PDB and (if small molecule) PDBQT.
    - Outputs preparation_report.html, preparation_report.json, and preparation_report.tsv.

    Args:
        input_pdb: Original input structure file path.
        output_dir: Output directory for prepared structures and reports.
        workflow_mode: 'small_molecule' or 'peptide'.
        decisions: Dict of reviewed preparation decisions.
        inspection_data: Dict of component inspection data.
        ph: Target pH for protonation.

    Returns:
        Dict with prepared paths, before/after statistics, and report files.
    """
    input_path = Path(input_pdb).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Original receptor PDB structure not found: {input_pdb}")

    out_dir = Path(ensure_dir(output_dir)).resolve()
    prepared_pdb_path = out_dir / f"{input_path.stem}_prepared.pdb"
    pdbqt_path = out_dir / "protein.pdbqt" if workflow_mode == "small_molecule" else None
    report_html_path = out_dir / "preparation_report.html"
    report_json_path = out_dir / "preparation_report.json"
    report_tsv_path = out_dir / "preparation_report.tsv"

    # Default decisions if not supplied
    if decisions is None:
        decisions = {}

    target_chains = decisions.get("selected_chains")
    water_policy = decisions.get("water_policy", "remove_bulk" if workflow_mode == "peptide" else "remove")
    if decisions.get("remove_water") is True and "water_policy" not in decisions:
        water_policy = "remove"
    elif decisions.get("remove_water") is False and "water_policy" not in decisions:
        water_policy = "keep"

    keep_metals = decisions.get("keep_metals", workflow_mode == "small_molecule")
    keep_cofactors = decisions.get("keep_cofactors", workflow_mode == "small_molecule")
    keep_ligands = set(r.upper() for r in decisions.get("keep_ligands", []))
    resolve_altlocs = decisions.get("resolve_altlocs", True)
    standardize_nonstandard = decisions.get("standardize_nonstandard", True)

    log_step(logger, f"Transparent Receptor Preparation ({workflow_mode})", "started", {
        "input": str(input_path),
        "output_dir": str(out_dir),
    })

    # Read original structure with gemmi
    st = gemmi.read_structure(str(input_path))
    if len(st) == 0:
        raise ValueError(f"No models found in PDB: {input_path.name}")

    orig_model = st[0]
    orig_res_count = sum(len(list(c)) for c in orig_model)
    orig_atom_count = sum(len(r) for c in orig_model for r in c)

    # Collect protein atom coordinates for interfacial water distance checks
    protein_atom_coords: list[tuple[float, float, float]] = []
    for chain in orig_model:
        if not target_chains or chain.name in target_chains:
            for res in chain:
                if res.entity_type == gemmi.EntityType.Polymer and res.name in STANDARD_AMINO_ACIDS:
                    for atom in res:
                        protein_atom_coords.append((atom.pos.x, atom.pos.y, atom.pos.z))

    def is_water_near_protein(wx: float, wy: float, wz: float, cutoff: float = 4.5) -> bool:
        if not protein_atom_coords:
            return False
        cutoff_sq = cutoff * cutoff
        for px, py, pz in protein_atom_coords:
            dx = wx - px
            dy = wy - py
            dz = wz - pz
            if (dx * dx + dy * dy + dz * dz) <= cutoff_sq:
                return True
        return False

    # Process structure into new model
    new_model = gemmi.Model("0")
    retained_metals = set()
    retained_cofactors = set()
    retained_ligands = set()
    retained_waters_count = 0
    removed_waters_count = 0
    altlocs_resolved_count = 0
    standardized_count = 0

    for chain in orig_model:
        if target_chains and chain.name not in target_chains:
            continue

        new_chain = gemmi.Chain(chain.name)

        for res in chain:
            rname = res.name.strip().upper()

            # Water molecules handling
            if res.entity_type == gemmi.EntityType.Water or rname in {"HOH", "WAT", "H2O", "TIP3", "DOD"}:
                if water_policy in ("remove", "remove_all"):
                    removed_waters_count += 1
                    continue
                elif water_policy in ("remove_bulk", "keep_interfacial"):
                    w_pos = res[0].pos if len(res) > 0 else None
                    if w_pos and is_water_near_protein(w_pos.x, w_pos.y, w_pos.z, cutoff=4.5):
                        retained_waters_count += 1
                    else:
                        removed_waters_count += 1
                        continue
                else:  # keep
                    retained_waters_count += 1

            # Non-polymer / heteroatom handling
            elif res.entity_type != gemmi.EntityType.Polymer and rname not in STANDARD_AMINO_ACIDS:
                if rname in KNOWN_METAL_IONS:
                    if not keep_metals and rname not in keep_ligands:
                        continue
                    retained_metals.add(rname)
                elif rname in KNOWN_COFACTORS:
                    if not keep_cofactors and rname not in keep_ligands:
                        continue
                    retained_cofactors.add(rname)
                elif rname in keep_ligands:
                    retained_ligands.add(rname)
                else:
                    # Unretained small molecule, additive, or buffer component
                    continue

            # Standardize non-standard amino acids (e.g. MSE -> MET)
            if standardize_nonstandard and rname in NONSTANDARD_AA_MAPPINGS:
                std_name = NONSTANDARD_AA_MAPPINGS[rname]
                res.name = std_name
                standardized_count += 1
                for atom in res:
                    if atom.name == "SE" and std_name == "MET":
                        atom.name = "SD"
                        atom.element = gemmi.Element("S")

            # Resolve alternate conformations (keep highest occupancy / first conformer)
            if resolve_altlocs:
                best_atoms: dict[str, gemmi.Atom] = {}
                for atom in res:
                    aname = atom.name
                    if aname not in best_atoms:
                        best_atoms[aname] = atom
                    else:
                        altlocs_resolved_count += 1
                        if atom.occ > best_atoms[aname].occ:
                            best_atoms[aname] = atom
                        elif atom.occ == best_atoms[aname].occ and best_atoms[aname].altloc != 'A' and atom.altloc == 'A':
                            best_atoms[aname] = atom

                clean_res = gemmi.Residue()
                clean_res.name = res.name
                clean_res.seqid = res.seqid
                clean_res.entity_type = res.entity_type
                for at in best_atoms.values():
                    at.altloc = '\0'  # clear altloc flag
                    clean_res.add_atom(at)
                new_chain.add_residue(clean_res)
            else:
                new_chain.add_residue(res)

        if len(new_chain) > 0:
            new_model.add_chain(new_chain)

    st_clean = gemmi.Structure()
    st_clean.add_model(new_model)
    st_clean.write_pdb(str(prepared_pdb_path))

    if not prepared_pdb_path.is_file() or prepared_pdb_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to write prepared receptor structure at {prepared_pdb_path}")

    # Read prepared structure to compute final counts
    st_prepared = gemmi.read_structure(str(prepared_pdb_path))
    prep_model = st_prepared[0]
    final_res_count = sum(len(list(c)) for c in prep_model)
    final_atom_count = sum(len(r) for c in prep_model for r in c)
    final_chains = [c.name for c in prep_model]

    # If small-molecule workflow, generate PDBQT file using obabel CLI
    if workflow_mode == "small_molecule" and pdbqt_path is not None:
        logger.info(f"Generating PDBQT format at {pdbqt_path} using obabel...")
        obabel_dir = os.path.dirname(sys.executable)
        obabel_bin = os.path.join(obabel_dir, "obabel")
        if not os.path.exists(obabel_bin):
            obabel_bin = "obabel"

        cmd = [obabel_bin, str(prepared_pdb_path), "-O", str(pdbqt_path), "-xr"]
        logger.debug(f"Executing: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("PDBQT file successfully created.")
        except Exception as e:
            logger.warning(f"Error during obabel PDBQT conversion: {e}. Retrying without -xr flag...")
            try:
                subprocess.run([obabel_bin, str(prepared_pdb_path), "-O", str(pdbqt_path)], capture_output=True, text=True, check=True)
            except Exception as e2:
                raise RuntimeError(f"Failed to generate docking PDBQT file from prepared receptor: {e2}") from e2

    # Build audit log of decisions
    decisions_log = []
    if inspection_data and inspection_data.get("review_items"):
        for it in inspection_data["review_items"]:
            user_act = decisions.get(it["id"], it["proposed_action"])
            decisions_log.append({
                "component": it["title"],
                "action": str(user_act),
                "scientific_rationale": it["scientific_rationale"],
                "docking_consequences": it["docking_consequences"],
            })
    else:
        decisions_log.append({
            "component": "Water Policy",
            "action": water_policy,
            "scientific_rationale": "Implicit solvent handling in docking potentials.",
            "docking_consequences": "Prevents artificial pocket blockage.",
        })
        decisions_log.append({
            "component": "Metal Ions",
            "action": "keep" if keep_metals else "remove",
            "scientific_rationale": "Maintains authentic active-site coordination.",
            "docking_consequences": "Affects electrostatic scoring.",
        })

    is_af = inspection_data.get("is_alphafold", False) if inspection_data else False
    struct_source = inspection_data.get("structure_source", "PDB Structure") if inspection_data else "PDB Structure"

    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "receptor_name": input_path.name,
        "workflow_mode": workflow_mode,
        "is_alphafold": is_af,
        "structure_source": struct_source,
        "alphafold_caveats": inspection_data.get("alphafold_caveats") if inspection_data else None,
        "original_path": str(input_path),
        "prepared_path": str(prepared_pdb_path),
        "pdbqt_path": str(pdbqt_path) if pdbqt_path else None,
        "report_html": str(report_html_path),
        "report_json": str(report_json_path),
        "report_tsv": str(report_tsv_path),
        "original_residues": orig_res_count,
        "prepared_residues": final_res_count,
        "original_atoms": orig_atom_count,
        "prepared_atoms": final_atom_count,
        "chains_selected": final_chains,
        "retained_metals": sorted(list(retained_metals)),
        "retained_cofactors": sorted(list(retained_cofactors)),
        "retained_ligands": sorted(list(retained_ligands)),
        "retained_waters_count": retained_waters_count,
        "removed_waters_count": removed_waters_count,
        "altlocs_resolved": altlocs_resolved_count,
        "standardized_residues": standardized_count,
        "decisions_log": decisions_log,
        "status": "completed",
    }

    # Write HTML report
    generate_html_preparation_report(report_data, report_html_path)

    # Write JSON report
    write_json(report_data, str(report_json_path))

    # Write TSV report
    with open(report_tsv_path, "w", encoding="utf-8") as f:
        f.write("Component\tAction\tScientific_Rationale\tDocking_Impact\n")
        for item in decisions_log:
            comp = item["component"].replace("\t", " ")
            act = item["action"].replace("\t", " ")
            rat = item["scientific_rationale"].replace("\t", " ").replace("\n", " ")
            imp = item["docking_consequences"].replace("\t", " ").replace("\n", " ")
            f.write(f"{comp}\t{act}\t{rat}\t{imp}\n")

    log_step(logger, f"Transparent Receptor Preparation ({workflow_mode})", "completed", {
        "original_pdb": str(input_path),
        "prepared_pdb": str(prepared_pdb_path),
        "report_html": str(report_html_path),
        "residues": f"{orig_res_count} -> {final_res_count}",
        "atoms": f"{orig_atom_count} -> {final_atom_count}",
    })

    return report_data
