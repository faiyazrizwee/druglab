# DrugLab 2.0

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-94%20passed-brightgreen.svg)](tests/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](Dockerfile)

**A lightweight, modular computational drug-discovery platform featuring dual docking pipelines:**
1. **Small-Molecule–Protein Docking & Optimization** powered by **AutoDock Vina**
2. **Protein–Peptide Docking** powered by **LightDock** (Glowworm Swarm Optimization)

DrugLab runs locally on standard research workstations and laptops without requiring GPUs, external cloud services, or paid platforms.

---

## Dual Docking Pipelines

```
DrugLab 2.0
├── Option 1: Small-Molecule Virtual Screening & Lead Optimization (AutoDock Vina)
│   ├── Target Protein Preparation (PDBFixer / PDBQT)
│   ├── Binding Pocket Detection & 3D Interactive Tuner
│   ├── Small-Molecule Ligand Preparation (RDKit / Meeko)
│   ├── Drug-Likeness Filters (Lipinski, Veber, PAINS, Brenk)
│   ├── ADMET Risk Scoring (ADMET-AI / RDKit)
│   ├── AutoDock Vina Docking & Pose Clustering
│   ├── Multi-Criteria Lead Selection (TOPSIS / Weighted Sum)
│   ├── Lead Optimization (Bioisosteres, Scaffold Hopping, Pareto Ranking)
│   └── Interactive Standalone HTML Report
│
└── Option 2: Protein–Peptide Docking (LightDock)
    ├── Target Receptor Ingestion & Validation
    ├── Protein Preparation specifically tailored for LightDock
    ├── Peptide Input (FASTA, Plain Sequence, or Existing PDB Structure)
    ├── Peptide Length & Size Filtering (Configurable thresholds + TSV audit)
    ├── Peptide ADMET & Physicochemical Profiling (pI, Net Charge, GRAVY, Instability, Boman index)
    ├── Peptide 3D Structural Modeling (ETKDGv3 + MMFF94 force-field minimization)
    ├── Binding-Site Selection & Restraints (Explicit, Pocket-guided, or Global search)
    ├── LightDock Docking Engine (Glowworm Swarm Optimization: fastdfire, dfire, cpydock, tobi)
    ├── Pose Structural Analysis (Clashes, Contacts, H-Bonds, Salt bridges, Interface residues)
    ├── Multi-Metric Candidate Ranking
    └── Standalone Interactive HTML Report with 3Dmol.js Complex Viewer
```

---

## Installation

### Option 1: Conda Environment (Recommended)

Conda easily manages cheminformatics and biophysical compiled C-extensions:

```bash
# Clone the repository
git clone https://github.com/faiyazrizwee/druglab.git
cd druglab

# Create and activate environment
conda env create -f environment.yml
conda activate druglab

# Install DrugLab core package in editable mode
pip install -e .

# Install optional Protein–Peptide Docking (LightDock) dependencies:
pip install -e ".[peptide]"

# Or install all optional components (ADMET-AI, OpenMM, LightDock, Dev):
pip install -e ".[all]"
```

### Option 2: Pip-Only Installation

```bash
# Install core dependencies
pip install -r requirements.txt

# Install DrugLab with peptide docking dependencies
pip install -e ".[peptide]"
```

### Option 3: Docker Container

A multi-platform containerized environment supporting both small-molecule and peptide workflows:

```bash
# Build standard image (Small-molecule workflow)
docker build -t druglab:latest .

# Build peptide-enabled image (LightDock included)
docker build --target peptide -t druglab:peptide .

# Run DrugLab interactive mode:
docker run -it --rm \
    -v $(pwd)/protein:/app/protein \
    -v $(pwd)/results:/app/results \
    druglab:peptide
```

---

## Quick Start & Usage

### 1. Interactive Dual-Pipeline Selector

Run `druglab` with no arguments in any interactive terminal to open the top-level workflow selector:

```bash
$ druglab

DrugLab 2.0
===========

Select docking workflow:

  1. Small-molecule–protein docking (Virtual screening & Lead optimization)
  2. Protein–peptide docking (LightDock engine)
  3. Exit

Enter choice [1-3] (default: 1):
```

### 2. Protein–Peptide Docking Workflow

#### Via `druglab run --mode peptide`

```bash
# Fast exploratory peptide docking from sequence
druglab run \
    --mode peptide \
    --protein protein/3DWT.pdb \
    --peptide-seq ACDEFGH \
    --preset fast \
    --output-dir results

# Batch virtual screening from a multi-peptide FASTA library (or directory of FASTA files)
druglab run \
    --mode peptide \
    --protein protein/3DWT.pdb \
    --peptide-fasta library_peptides.fasta \
    --admet-provider peplife \
    --preset fast \
    --output-dir results
```

#### Via Dedicated `druglab peptide` Modular Subcommands

```bash
# 1. Full pipeline run (Protein Prep → Peptide Prep → Pocket → ADMET → Docking → Leads)
druglab peptide run \
    --protein protein/3DWT.pdb \
    --sequence WYVIK \
    --preset fast \
    --output-dir results

# 2. Receptor protein inspection & preparation for LightDock
druglab peptide protein-prepare \
    --protein protein/3DWT.pdb \
    --chains A \
    --water-policy remove_bulk \
    --output-dir results/prepared_proteins

# 3. Peptide ligand preparation (zwitterion default, capped ACE/NHE, cyclic)
druglab peptide prepare \
    --sequence ACDEFGH \
    --n-term zwitterion \
    --c-term zwitterion \
    --output-dir results/prepared_peptides

# 4. Cavity detection & interactive 3D box tuner GUI → LightDock restraints
druglab peptide pocket \
    --protein results/prepared_proteins/3DWT_prepared.pdb \
    --auto-confirm \
    --output-dir results/pocket

# 5. Peptide ADMET, barrier permeability & safety screening
druglab peptide admet \
    --sequence "ACDEFGH,CYIQNCPLG" \
    --provider local \
    --output-dir results/admet

# 6. LightDock docking execution
druglab peptide dock \
    --protein results/prepared_proteins/3DWT_prepared.pdb \
    --peptide results/prepared_peptides/peptide_peptide_1.pdb \
    --restraints results/pocket/restraints.list \
    --preset fast \
    --output-dir results/docking

# 7. Multi-objective lead ranking & triage
druglab peptide select-leads \
    --run-dir results/peptide_run_20261029_214703 \
    --top-n 5

# 8. Regenerate interactive HTML report
druglab peptide report \
    --input results/peptide_run_20261029_214703 \
    --output report.html
```

### 3. Small-Molecule Virtual Screening & Lead Optimization

100% backward compatible with existing DrugLab pipelines:

```bash
# Default mode is small-molecule
druglab run \
    --protein protein/4MZI.pdb \
    --library ligands/ \
    --output-dir output

# Or explicitly specified
druglab run \
    --mode small-molecule \
    --protein protein/4MZI.pdb \
    --library ligands/ \
    --output-dir output
```

---

## Peptide Docking Technical Architecture

### 1. Peptide Structural Modeling
Peptide sequences are constructed using RDKit, parameterized with hydrogens, embedded in 3D using ETKDGv3, and energy-minimized using the **MMFF94** force field. Heavy-atom PDB structures with standard stereochemistry and atom naming are produced for LightDock. Pre-existing validated PDB structures can also be provided directly via `--structure`.

### 2. Peptide ADMET, Permeability & Safety Profiling
- **Local Peptide ADMET Engine (Default)**: Comprehensive, offline-capable peptide profiling engine built on RDKit and BioPython ProtParam:
  - **Calculated Descriptors**: Molecular Weight (MW), Wildman-Crippen LogP, H-Bond Donors (HBD), H-Bond Acceptors (HBA), and Topological Polar Surface Area (TPSA).
  - **Reference Library Similarity**: MACCS-166 Tanimoto fingerprint similarity against a configurable clinical peptide reference panel (Octreotide, Leuprolide, Oxytocin, Vasopressin, Bivalirudin, Exenatide, Glucagon, Pramlintide, Ziconotide, Semaglutide).
  - **Biological Half-Life**: Predicted in vitro/in vivo half-life based on N-terminal degradation rules and Guruprasad instability indices.
  - **Membrane Permeability Models**: Literature-validated permeability predictors (PMID 39031123, 40134508, 9090704) covering Caco-2 (linear, cyclic, and all-peptide models), PAMPA for cyclic peptides, and RRCK for cyclic peptides. Unavailable/inapplicable endpoints clearly reported as `N/A`.
  - **Blood-Brain Barrier (BBB)**: Qualitative BBB penetration prediction (`BBB+` / `BBB-`) using SCMB3PP dipeptide propensities and BBB shuttle motif detection (PMID 36284036, 38637801).
  - **Peptide-Specific Toxicity Assessment**: Validated hemolytic and membrane-disruptive risk assessment (PMID 32145017, 39905233) without applying small-molecule toxicity classifiers; reports `toxicity undetermined` when outside the validated domain.
- **PEPLife / PEPLife 2.0**: Specialized peptide biological half-life and proteolytic degradation database profiling enriched with local descriptor models.
- **Flexible Peptide Lengths & Diagnostics**: The rigid length filter is removed to allow docking of peptides of any length. Advisory warnings are displayed on the terminal, diagnostic logs, and interactive HTML reports whenever sequence length falls outside the recommended range (5–30 AA).
- **Provider Selection**: Select via interactive prompt or CLI flags (`--admet-provider {local,peplife,none}`).

### 3. Binding-Site Modes & LightDock Restraints
- **`explicit`**: Validates user-supplied residue identifiers (e.g. `A.TYR.45`) against the prepared receptor structure and writes `restraints.list`.
- **`pocket_guided`**: Maps DrugLab geometric pocket cavities or contact residues from a co-crystallized reference ligand onto the receptor surface.
- **`global`**: Unrestrained surface sampling across the entire receptor using Glowworm Swarm Optimization.

### 4. Sampling Presets
| Preset | Swarms | Simulation Steps | Glowworms / Swarm | Intended Use |
|---|---|---|---|---|
| `fast` | 10 | 10 | 50 | Rapid smoke tests & exploratory screening |
| `balanced` | 30 | 25 | 50 | Routine candidate comparison |
| `thorough` | 100 | 50 | 100 | In-depth candidate pose refinement |

Supported scoring functions include `fastdfire` (default), `dfire`, `cpydock`, and `tobi`.

### 5. Interactive Standalone HTML Report
Both the small-molecule (`summary.html`) and peptide docking reports (`report.html`) are fully self-contained and feature:
- **Executive KPI Cards**: Receptor target, total candidates screened, filter attrition, and best binding affinities / docking scores.
- **Interactive 3D Complex Viewer**: Powered by **3Dmol.js** with cartoon, surface, ribbon, stick, and wireframe representation controls, palette selectors, camera zoom (`Focus Ligand` / `Focus Peptide`), and pose selection.
- **Pure Shape-Based Hover Interactivity Engine**:
  - Hovering any amino acid residue or ligand atom immediately renders a prominent glowing ball-and-stick highlight (`#f6e05e`) directly in 3D without tearing down WebGL geometry.
  - Automatically classifies non-covalent interactions (Salt Bridge, Hydrogen Bond, Cation–$\pi$, $\pi$–$\pi$ Stacking, Hydrophobic, Van der Waals, and Steric Clashes) with distance in Å.
  - Draws a pulsing partner sphere and a dashed 3D connector cylinder between contacting heavy atoms.
  - Displays an in-canvas 3D floating label and updates a sleek bottom HUD banner with residue names, interaction badges, and partner details.
- **Ranked Results & Pose Metrics Table**: Binding energies, inter-chain contacts (<4.5 Å), hydrogen bonds (<3.5 Å), steric clashes (<2.0 Å), and safety liabilities.
- **ADMET & Filtering Diagnostics**: Complete audit log recording length, molecular weight, predicted permeability, and stability.
- **Run Manifest**: Cryptographic checksums of inputs, software versions, random seeds, and parameters for full scientific reproducibility.

---

## Configuration Reference (`configs/peptide_docking.yaml`)

```yaml
workflow:
  mode: peptide

protein:
  input: null
  chains: []
  remove_waters: true
  remove_hydrogens: false
  keep_heteroatoms: false
  ph: 7.4

peptide:
  fasta: null
  sequence: null
  structure: null
  allow_nonstandard_residues: false
  conformers: 1
  minimize: true
  forcefield: "MMFF94"
  seed: 42

filtering:
  enabled: true
  min_length_aa: 5
  max_length_aa: 30
  max_mw: 4000.0
  retain_filtered_candidates: true

admet:
  enabled: true
  provider: "local"
  on_prediction_failure: "warn_and_continue"

binding_site:
  mode: "auto"
  receptor_residues: []
  peptide_residues: []
  restraints_enabled: false
  pocket_method: "geometric"
  pocket_cutoff_distance: 6.0

lightdock:
  executable: null
  preset: "fast"
  scoring_function: "fastdfire"
  swarms: null
  glowworms: 50
  steps: null
  anm: false
  timeout_seconds: 3600
  seed: 42

resources:
  max_parallel_jobs: 1
  max_cpu_processes: 4

report:
  html: true
  interactive_3d: true
  title: "DrugLab Peptide Docking Report"
```

---

## Automated Testing

DrugLab maintains 94 automated tests across unit and integration categories:

```bash
# Run fast default unit tests (excluding real integration tests)
pytest -m "not integration"

# Run real LightDock integration smoke test
pytest -m integration

# Run full test suite with test coverage
pytest tests/ -v --cov=core --cov=drug_selection --cov=lead_optimization --cov=utils --cov=reports
```

---

## Scientific Caveats & Limitations

1. **Scoring Potentials vs. True Affinity**: LightDock scores (DFIRE, CPYDOCK, TOBI) are statistical or empirical interaction potentials. They do not equal experimentally measured thermodynamic binding constants ($K_d$ or $\Delta G$).
2. **Peptide Conformational Flexibility**: High peptide flexibility is modeled using conformer ensembles and Glowworm Swarm Optimization; however, long or disordered peptides (>30 AA) may adopt solution ensembles beyond rigid or semi-flexible docking models.
3. **Computational ADMET Predictions**: Predicted instability indices, Boman indices, and GRAVY hydropathicity values are sequence-derived empirical metrics and should be treated as guidance rather than clinical or experimental certainty.

---

## Citations & Acknowledgments

- **LightDock**: Jiménez-García B, Pons C, Fernández-Recio J. *LightDock: a new multi-scale approach to protein–protein docking.* Bioinformatics (2018).
- **AutoDock Vina**: Eberhardt J, Santos-Martins D, Tillack AF, Forli S. *AutoDock Vina 1.2.0: New Docking Methods, Expanded Force Field, and Python Bindings.* J. Chem. Inf. Model. (2021).
- **RDKit**: Landrum G, et al. *RDKit: Open-source cheminformatics.*
- **3Dmol.js**: Rego N, Koes D. *3Dmol.js: molecular visualization without Java.* Bioinformatics (2015).

---

## License

DrugLab is licensed under the [MIT License](LICENSE).
