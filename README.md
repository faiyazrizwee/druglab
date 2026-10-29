# DrugLab

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-45%20passed-brightgreen.svg)](tests/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](Dockerfile)

**A lightweight, modular command-line platform for computational drug discovery, virtual screening, lead identification, and lead optimization.**

DrugLab runs locally on standard research workstations and laptops without requiring GPUs, external cloud services, or paid web platforms.

---

## Features

### Drug Selection Workflow
Identify and rank promising drug leads from a chemical library against a target protein:
- **Protein Preparation** — Clean PDB structures, resolve missing residues/atoms with PDBFixer, add hydrogens at target pH, and convert to PDBQT.
- **Binding Pocket Detection & Interactive 3D Tuner** — Automatic geometric cavity detection (DBSCAN), reference-ligand bounding, manual coordinate entry, or an **Interactive 3D Pocket Tuner Web GUI** with real-time box sizing up to 60.0 Å.
- **Ligand Preparation** — Validate, sanitize, generate low-energy 3D conformers with RDKit, assign Gasteiger charges, and format to PDBQT via Meeko.
- **Drug-Likeness Screening** — Comprehensive rule filters: Lipinski's Rule of 5, Veber, Ghose, Egan, Muegge, PAINS, and Brenk structural alerts.
- **ADMET Screening** — Fast absorption, distribution, metabolism, excretion, and toxicity risk profiling (via ADMET-AI or RDKit QSAR fallback).
- **Molecular Docking** — AutoDock Vina binding affinity calculations with flexible grid exhaustiveness and pose clustering.
- **Lead Selection** — Multi-criteria composite ranking (Weighted Sum or TOPSIS) incorporating binding affinity, Ligand Efficiency (LE), Lipophilic Efficiency (LipE), Synthetic Accessibility (SAScore), and ADMET safety scores.

### Lead Optimization Workflow
Optimize selected lead molecules into potent drug candidates:
- **Analogue Generation** — Bioisosteric replacement, scaffold hopping, and R-group enumeration.
- **Multi-Objective Evaluation** — Simultaneous re-docking, drug-likeness checks, and ADMET profiling.
- **Candidate Ranking** — Pareto frontier optimization (via pymoo) to balance potency against pharmacokinetic liabilities.

### Interactive Reporting & Visualization
- **Clean Box-Style HTML Reports** — Standalone, shareable `reports/summary.html` with an executive scorecard, compound attrition funnel, and 2D vector structures.
- **Interactive 3D Molecular Viewer** — High-contrast 3Dmol.js canvas supporting Dark/Light background switching, pose-by-pose $\Delta G$ / LE / LipE display, multi-style protein and ligand rendering (Cartoon, Sticks, Lines, Surface, Ball & Stick), and camera focusing on docked poses.
- **Kernel & Browser Memory Safety** — Constrained surface computations and explicit WebGL context disposal to prevent memory leaks.
- **Interactive Analytics** — Plotly charts with high-contrast grid lines for chemical space exploration (MW vs. cLogP) and binding energy distributions.
- **Quarantine Audit** — Resilient batch error handling logging corrupted compounds into `quarantine.json` without failing downstream execution.

---

## Installation

### Option 1: Conda Environment (Recommended)

Conda easily manages complex cheminformatics C-extensions (RDKit, AutoDock Vina, Open Babel, OpenMM):

```bash
# Clone the repository
git clone https://github.com/druglab/druglab.git
cd druglab

# Create and activate environment
conda env create -f environment.yml
conda activate druglab

# Install DrugLab CLI package in editable mode
pip install -e .
```

### Option 2: Pip Installation

If you already manage your scientific packages in a virtual environment:

```bash
# Clone the repository
git clone https://github.com/druglab/druglab.git
cd druglab

# Install core dependencies
pip install -r requirements.txt

# Install DrugLab package in editable mode
pip install -e .

# Or install all optional extensions (ADMET-AI, OpenMM, simulation, dev tools):
pip install -e ".[all]"
```

### Option 3: Docker Container

A multi-platform containerized environment pre-configured with all dependencies:

```bash
# Build the Docker image
docker build -t druglab:latest .

# Run DrugLab via Docker with volume mounts and port forwarding for the 3D Tuner GUI:
docker run -it --rm \
    -p 8080:8080 \
    -v $(pwd)/protein:/app/protein \
    -v $(pwd)/ligands:/app/ligands \
    -v $(pwd)/output:/app/output \
    druglab:latest run \
    --protein protein/4MZI.pdb \
    --library ligands/ \
    --output-dir output
```

### Verify Installation

```bash
druglab --help
```

---

## Quick Start

### 1. Run Complete Virtual Screening & Lead Selection

```bash
druglab run \
    --protein protein/4MZI.pdb \
    --library ligands/ \
    --output-dir output
```

This runs the end-to-end pipeline:
1. Protein Preparation (PDB cleaning, hydrogen addition, PDBQT conversion)
2. Binding Pocket Detection (Cavity clustering or interactive selection)
3. Ligand Preparation (3D conformer generation, charge assignment)
4. Drug-Likeness Screening (Lipinski, Veber, PAINS, Brenk)
5. ADMET Screening (Pharmacokinetic & toxicity profiling)
6. Molecular Docking (AutoDock Vina)
7. Lead Selection & Composite Ranking
8. Interactive HTML Report Generation (`output/reports/summary.html`)

### 2. Run Lead Optimization

```bash
druglab optimize \
    --lead output/lead_selection/leads.sdf \
    --protein output/protein_preparation/protein.pdbqt \
    --box output/pocket_detection/docking_box.json \
    --output-dir output
```

---

## Interactive 3D Pocket & Docking Box Tuner

DrugLab features an in-browser 3D docking box tuner that launches directly from the CLI.

```bash
# Launch interactive 3D tuner on a prepared protein
druglab pocket detect \
    --protein output/protein_preparation/protein_clean.pdb \
    --gui
```

When running the interactive pipeline (`druglab run`), you can also select option `[g]` at the pocket selection prompt to open the GUI tuner.

### Key Capabilities:
- **Real-Time 3D Cavity Visualization**: Protein backbone and cavity residues rendered in high resolution.
- **Center Coordinates (X, Y, Z)**: Fine-tune center positions with slider controls, number inputs, or step buttons (`+1/-1`).
- **Docking Box Dimensions (Size X, Y, Z)**: Adjust bounding box size up to **60.0 Å** with instant 3D box boundary updates.
- **Protein & Ligand Styles**: Switch between Cartoon, Sticks, Lines/Wireframe, and Surface.
- **One-Key Confirmation**: Press <kbd>Enter</kbd> (or click **Confirm Coordinates**) to lock the box, shut down the local server, and pass coordinates directly into downstream docking.

---

## Interactive HTML Summary Reports

After docking and lead selection, DrugLab produces `output/reports/summary.html`:

- **3D Pose Viewer**:
  - Switch between **Dark Mode** and **Light Mode** backgrounds.
  - Browse every docked ligand and all binding poses.
  - Real-time binding metric overlays: $\Delta G$ (kcal/mol), Ligand Efficiency (LE), and Lipophilic Efficiency (LipE).
  - Protein style selector: Cartoon, Sticks, Lines/Wireframe, Active-Site Surface.
  - Ligand style selector: Sticks, Ball & Stick, Spacefill, Wireframe.
  - **Focus Docked Ligand** button to zoom directly into the binding pocket.
  - Clean box-style interface with zero distracting emoji clutter.
- **Medicinal Chemistry Analytics**: Plotly-powered charts with high-contrast grid lines for Chemical Space (MW vs. cLogP), Affinity Distributions, and Lead Comparison Radar profiles.
- **Interactive Leads Table**: Filterable and sortable table with inline 2D SVG molecular structures.

To regenerate or recompile reports at any time:

```bash
druglab report --input output/
```

---

## Command Line Reference

Every module can be run independently or composed as part of a custom pipeline.

### Protein Preparation
```bash
druglab protein prepare \
    --input protein/7DTD.pdb \
    --chain A \
    --ph 7.4 \
    --output-dir output
```

### Pocket Detection
```bash
# Automated geometric cavity detection
druglab pocket detect --protein output/protein_preparation/protein_clean.pdb

# Interactive 3D Web GUI Tuner
druglab pocket detect --protein output/protein_preparation/protein_clean.pdb --gui

# Ligand-guided pocket detection
druglab pocket detect \
    --protein output/protein_preparation/protein_clean.pdb \
    --method ligand_based \
    --reference-ligand reference_ligand.sdf

# Manual coordinates
druglab pocket detect \
    --protein output/protein_preparation/protein_clean.pdb \
    --method manual \
    --center 110.983 109.918 108.921 \
    --size 40.0 40.0 40.0
```

### Ligand Preparation
```bash
# Prepare a directory of SDF files or a single multi-mol SDF
druglab ligand prepare --input ligands/ --output-dir output
```

### Drug-Likeness Screening
```bash
druglab screen druglikeness \
    --input output/ligand_preparation/prepared_ligands.sdf \
    --rules lipinski veber pains brenk \
    --output-dir output
```

### ADMET Screening
```bash
druglab screen admet \
    --input output/druglikeness/passed_ligands.sdf \
    --output-dir output
```

### Molecular Docking
```bash
druglab dock \
    --protein output/protein_preparation/protein.pdbqt \
    --ligands output/admet/passed_ligands.sdf \
    --box output/pocket_detection/docking_box.json \
    --exhaustiveness 16 \
    --n-poses 9 \
    --output-dir output
```

### Lead Selection
```bash
druglab select \
    --docking-results output/docking/docking_results.csv \
    --admet-results output/admet/admet_results.csv \
    --druglikeness-results output/druglikeness/descriptors.csv \
    --scoring-method weighted_sum \
    --top-n 5 \
    --output-dir output
```

### Data Retrieval (`druglab fetch`)
```bash
# Fetch crystal structure PDB from RCSB
druglab fetch pdb 7DTD --output protein/7DTD.pdb

# Fetch ligand SDF by name from PubChem
druglab fetch ligand "aspirin" --output ligands/aspirin.sdf

# Fetch ligand SDF by PubChem CID
druglab fetch cid 2244 --output ligands/aspirin.sdf
```

### Stop-and-Resume Workflow
```bash
# Run up to ligand preparation and pause
druglab run --protein protein/7DTD.pdb --library ligands/ --stop-after ligand_preparation

# Resume from drug-likeness through ADMET
druglab run --protein protein/7DTD.pdb --library ligands/ --resume druglikeness --stop-after admet
```

---

## Configuration System

DrugLab uses a hierarchical configuration system:

```
CLI Flags  →  User YAML Config  →  Default YAML Config  →  Code Defaults
```

Save custom configurations in YAML:

```yaml
project:
  name: "custom_virtual_screen"
  output_dir: "output"

protein_preparation:
  remove_water: true
  add_hydrogens: true
  ph: 7.4
  chain_id: "A"

pocket:
  method: "geometric"
  padding: 10.0
  max_box_dimension: 60.0

docking:
  exhaustiveness: 16
  n_poses: 9
  seed: 42

druglikeness:
  rules:
    - lipinski
    - veber
    - pains
    - brenk
  max_violations: 1

lead_selection:
  top_n: 5
  scoring_method: "weighted_sum"
  weights:
    affinity: 0.45
    admet: 0.25
    druglikeness: 0.15
    efficiency: 0.15
```

Pass your config file with `--config`:
```bash
druglab run --config configs/my_config.yml
```

---

## Project Structure

```
druglab/
├── druglab.py                  # CLI entry point
├── pyproject.toml              # Modern PEP 517/621 packaging metadata
├── requirements.txt            # Python dependencies
├── environment.yml             # Conda environment definition
├── Dockerfile                  # Containerized deployment
├── configs/                    # YAML configuration files
│   └── default.yml
├── core/                       # Computational engines
│   ├── protein_preparation.py  # PDB cleaning & PDBQT conversion
│   ├── pocket_detection.py     # Binding site detection (DBSCAN / manual)
│   ├── pocket_gui.py           # Interactive 3D Pocket Tuner GUI (3Dmol.js)
│   ├── ligand_preparation.py   # Ligand conformer generation & PDBQT prep
│   ├── druglikeness.py         # Molecular property rules & filters
│   ├── admet.py                # ADMET predictions & risk profiling
│   ├── docking.py              # AutoDock Vina wrapper & pose parsing
│   └── workflow_engine.py      # Module execution orchestrator
├── drug_selection/             # Workflow 1: Drug Selection
│   ├── lead_selection.py       # Multi-criteria lead ranking (TOPSIS / Weighted Sum)
│   └── workflow.py             # Pipeline execution manager
├── lead_optimization/          # Workflow 2: Lead Optimization
│   ├── analogue_generation.py  # Bioisosteres & scaffold hopping
│   ├── screening.py            # Analogue screening pipeline
│   ├── docking.py              # Analogue docking wrapper
│   └── ranking.py              # Pareto multi-objective ranking
├── reports/                    # Interactive Report Generation
│   └── generator.py            # HTML summary report & Plotly charts
├── utils/                      # Shared utilities
│   ├── chemistry.py            # SAScore, LE, LLE/LipE calculations
│   ├── config.py               # YAML configuration loader
│   ├── io.py                   # Atomic file I/O & JSON serialization
│   └── logger.py               # Rich terminal output formatting
└── tests/                      # Automated test suite (45 unit & e2e tests)
```

---

## Dependencies

| Package | Purpose |
|---|---|
| **RDKit** | Cheminformatics, conformer generation, descriptor calculations |
| **AutoDock Vina** | Molecular docking engine and affinity evaluation |
| **Meeko** | RDKit molecule conversion to AutoDock PDBQT |
| **Gemmi** | Crystallographic structure manipulation and PDB parsing |
| **Biopython** | Macromolecular structure analysis |
| **PDBFixer / OpenMM** | Protein structure repair, missing heavy atoms, and hydration |
| **Open Babel** | Chemical format translation |
| **ADMET-AI** | Machine learning ADMET property predictions |
| **Plotly** | High-contrast interactive 2D charts for reports |
| **3Dmol.js** | Browser-based WebGL 3D molecular visualization |
| **scikit-learn** | DBSCAN pocket clustering and score normalization |
| **pymoo** | Multi-objective Pareto optimization |
| **Rich** | Terminal formatting, progress indicators, and status tables |
| **Matplotlib** | Static report and analytics figure generation |
| **PyYAML** | Hierarchical workflow configuration |

---

## Testing

DrugLab maintains a comprehensive test suite covering all computational engines and workflows:

```bash
# Run the complete test suite
pytest tests/ -v

# Run with test coverage
pytest tests/ -v --cov=core --cov=drug_selection --cov=lead_optimization --cov=utils --cov=reports
```

All 45 unit and integration tests pass reliably.

---

## License

DrugLab is licensed under the [MIT License](LICENSE).
