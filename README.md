# DrugLab

**A lightweight command-line platform for computational drug discovery, virtual screening, lead identification, and lead optimization.**

DrugLab runs locally on standard research computers (16 GB RAM laptops) without requiring GPUs, cloud services, or web servers.

---

## Features

### Drug Selection Workflow

Identify promising lead molecules from a ligand library against a protein target.

- **Protein Preparation** — Clean PDB, fix missing residues/atoms, add hydrogens, convert to PDBQT
- **Binding Pocket Detection** — Geometric cavity detection, ligand-based, or manual box definition
- **Ligand Preparation** — Sanitize, generate 3D conformers, compute charges, convert to PDBQT
- **Drug-Likeness Screening** — Lipinski, Veber, Ghose, Egan, Muegge, PAINS, Brenk filters
- **ADMET Screening** — Absorption, Distribution, Metabolism, Excretion, Toxicity prediction
- **Molecular Docking** — AutoDock Vina binding affinity calculation
- **Lead Selection** — Multi-criteria ranking with composite scoring

### Lead Optimization Workflow

Optimize a lead molecule by generating and evaluating structural analogues.

- **Analogue Generation** — Bioisosteric replacement, scaffold hopping, R-group enumeration
- **Drug-Likeness Evaluation** — Filter analogues by drug-likeness rules
- **ADMET Evaluation** — Predict pharmacokinetic and safety properties
- **Docking Validation** — Compare analogue binding affinities to original lead
- **Candidate Ranking** — Multi-objective ranking of optimized candidates

### Design Principles

- **Local execution** — No cloud, no GPU, no internet required at runtime
- **Reproducible workflows** — Config snapshots saved with every run
- **Modular architecture** — Every module runs independently
- **Fully automated** — Run entire pipeline with a single command
- **Step-by-step** — Stop after any stage, inspect, modify, and resume
- **Manual parameter control** — Override any parameter via CLI or config file
- **Extensible** — Clean interfaces for future ML models and MD simulations

---

## Installation

### Using Docker

Build the Docker image locally:

```bash
docker build -t druglab:latest .
```

Run DrugLab using Docker (mount your directories so the container can read and write files):

```bash
docker run --rm \
    -v $(pwd)/protein:/app/protein \
    -v $(pwd)/ligands:/app/ligands \
    -v $(pwd)/output:/app/output \
    druglab:latest run \
    --protein protein/4MZI.pdb \
    --library ligands/ \
    --output-dir output
```

### Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate druglab
```

### Using pip

```bash
pip install -r requirements.txt
```

### Verify Installation

```bash
python druglab.py --help
```

---

## Quick Start

### Run Complete Drug Selection

```bash
python druglab.py run \
    --protein protein/4MZI.pdb \
    --library ligands/ \
    --output-dir output
```

This runs the full pipeline: protein preparation → pocket detection → ligand preparation → drug-likeness → ADMET → docking → lead selection.

### Run Complete Lead Optimization

```bash
python druglab.py optimize \
    --lead output/lead_selection/leads.sdf \
    --protein output/protein_preparation/protein.pdbqt \
    --box output/pocket_detection/docking_box.json \
    --output-dir output
```

---

## Usage

### Individual Modules

Every module can be run independently. Outputs are saved as files that can be inspected, modified, and fed into subsequent stages.

#### Protein Preparation

```bash
python druglab.py protein prepare \
    --input protein/7DTD.pdb \
    --chain A \
    --ph 7.4 \
    --output-dir output
```

Output: `output/protein_preparation/protein_clean.pdb`, `protein.pdbqt`

#### Pocket Detection

```bash
# Automatic geometric detection
python druglab.py pocket detect \
    --protein output/protein_preparation/protein_clean.pdb

# Using a reference ligand
python druglab.py pocket detect \
    --protein output/protein_preparation/protein_clean.pdb \
    --method ligand_based \
    --reference-ligand reference_ligand.sdf

# Manual box definition
python druglab.py pocket detect \
    --protein output/protein_preparation/protein_clean.pdb \
    --method manual \
    --center 12.3 45.6 78.9 \
    --size 20.0 20.0 20.0
```

Output: `output/pocket_detection/docking_box.json`, `pockets.json`

#### Ligand Preparation

```bash
# From directory of SDF files
python druglab.py ligand prepare \
    --input ligands/

# From single SDF file
python druglab.py ligand prepare \
    --input ligands/apigenin.sdf
```

Output: `output/ligand_preparation/prepared_ligands.sdf`, `ligands_pdbqt/`

#### Drug-Likeness Screening

```bash
python druglab.py screen druglikeness \
    --input output/ligand_preparation/prepared_ligands.sdf \
    --rules lipinski veber pains
```

Output: `output/druglikeness/passed_ligands.sdf`, `descriptors.csv`

#### ADMET Screening

```bash
python druglab.py screen admet \
    --input output/druglikeness/passed_ligands.sdf
```

Output: `output/admet/passed_ligands.sdf`, `admet_results.csv`

#### Molecular Docking

```bash
python druglab.py dock \
    --protein output/protein_preparation/protein.pdbqt \
    --ligands output/admet/passed_ligands.sdf \
    --box output/pocket_detection/docking_box.json \
    --exhaustiveness 32 \
    --n-poses 10
```

Output: `output/docking/docking_results.csv`, `poses/`, `best_poses.sdf`

#### Lead Selection

```bash
python druglab.py select \
    --docking-results output/docking/docking_results.csv \
    --admet-results output/admet/admet_results.csv \
    --druglikeness-results output/druglikeness/descriptors.csv \
    --top-n 5
```

Output: `output/lead_selection/leads.sdf`, `ranking.csv`

### Stop-and-Resume Workflows

Run the pipeline up to a specific stage:

```bash
python druglab.py run \
    --protein protein/7DTD.pdb \
    --library ligands/ \
    --stop-after ligand_preparation
```

Inspect outputs, modify parameters, then resume:

```bash
python druglab.py run \
    --protein protein/7DTD.pdb \
    --library ligands/ \
    --resume druglikeness
```

Run a specific range of stages:

```bash
python druglab.py run \
    --protein protein/7DTD.pdb \
    --library ligands/ \
    --resume druglikeness \
    --stop-after admet
```

### Reports & Output Data

Reports (such as `docking_report.json` and `lead_selection_report.json`) are automatically saved as JSON files in their respective output subdirectories. A unified HTML report generator CLI is planned for v0.2.

### Configuration

Override defaults with a YAML config file:

```bash
python druglab.py run \
    --protein protein/7DTD.pdb \
    --library ligands/ \
    --config configs/my_config.yml
```

See `configs/default.yml` for all available options.

---

## Configuration

DrugLab uses a layered configuration system:

```
CLI flags  →  User config YAML  →  Default config YAML  →  Hardcoded defaults
```

Higher priority overrides lower. Every run saves a `config_snapshot.yml` for reproducibility.

### Example Config

```yaml
project:
  name: "gaba_a_screening"
  output_dir: "output"

protein_preparation:
  remove_water: true
  add_hydrogens: true
  ph: 7.4
  chain_id: "A"

pocket_detection:
  method: "geometric"
  padding: 10.0

docking:
  exhaustiveness: 32
  n_poses: 10
  seed: 42

druglikeness:
  rules:
    - lipinski
    - veber
    - pains
  max_violations: 1

lead_selection:
  top_n: 5
  scoring_method: "weighted_sum"
  weights:
    affinity: 0.5
    admet: 0.3
    druglikeness: 0.2
```

---

## Architecture

```
druglab/
├── druglab.py                  # CLI entry point
├── configs/                    # Configuration files
│   └── default.yml
├── core/                       # Computational engines
│   ├── protein_preparation.py  # PDB cleaning & PDBQT conversion
│   ├── pocket_detection.py     # Binding site detection
│   ├── ligand_preparation.py   # Ligand prep & conformer generation
│   ├── druglikeness.py         # Rule-based molecular filters
│   ├── admet.py                # ADMET-AI predictions
│   └── docking.py              # AutoDock Vina docking
├── drug_selection/             # Workflow 1: Drug Selection
│   ├── lead_selection.py       # Multi-criteria lead ranking
│   └── workflow.py             # Pipeline orchestrator
├── lead_optimization/          # Workflow 2: Lead Optimization
│   ├── analogue_generation.py  # Structural analogue generation
│   ├── screening.py            # Combined DL + ADMET screening
│   ├── docking.py              # Analogue docking wrapper
│   ├── admet.py                # Optimization-specific ADMET
│   ├── ranking.py              # Final candidate ranking
│   └── workflow.py             # Pipeline orchestrator
├── utils/                      # Shared utilities
│   ├── logger.py               # Rich logging
│   ├── config.py               # Config loading & validation
│   ├── io.py                   # File I/O helpers
│   └── chemistry.py            # Shared chemistry utilities
├── reports/                    # Report generation
│   └── generator.py            # HTML/CSV/JSON reports
├── tests/                      # Test suite
├── examples/                   # Usage examples
├── protein/                    # Input protein files
└── ligands/                    # Input ligand library
```

### Workflow 1: Drug Selection

```
Protein (PDB)  ──→  Protein Preparation  ──→  Pocket Detection
                                                      │
Ligand Library ──→  Ligand Preparation  ──→  Drug-Likeness  ──→  ADMET
                                                                    │
                              Lead Selection  ←──  Docking  ←───────┘
                                    │
                              leads.sdf + ranking.csv
```

### Workflow 2: Lead Optimization

```
Lead Molecule  ──→  Analogue Generation  ──→  Screening (DL + ADMET)
                                                      │
Protein + Box  ──→  Docking  ←────────────────────────┘
                       │
                   Ranking  ──→  top_candidates.sdf
```

---

## Output Structure

```
output/
├── config_snapshot.yml              # Reproducibility record
├── logs/                            # Execution logs
├── protein_preparation/             # Cleaned protein files
├── pocket_detection/                # Detected pockets & docking box
├── ligand_preparation/              # Prepared ligands & PDBQT files
├── druglikeness/                    # Screening results & descriptors
├── admet/                           # ADMET predictions
├── docking/                         # Docking scores & poses
├── lead_selection/                  # Final leads & rankings
├── analogue_generation/             # Generated analogues
├── screening/                       # Analogue screening results
├── analogue_docking/                # Analogue docking results
├── ranking/                         # Final optimization ranking
└── reports/                         # HTML/CSV/JSON reports
```

---

## Dependencies

| Package | Purpose |
|---|---|
| RDKit | Cheminformatics, descriptors, molecular manipulation |
| Biopython | PDB parsing and structure manipulation |
| PDBFixer | Missing residue and atom repair |
| Meeko | RDKit-to-PDBQT conversion for AutoDock |
| Gemmi | Crystallographic data and PDB processing |
| Open Babel | Molecular format conversion |
| AutoDock Vina | Molecular docking engine |
| ADMET-AI | ADMET property prediction |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| scikit-learn | Score normalization and clustering |
| Rich | Terminal output formatting and logging |
| Plotly | Interactive charts for reports |
| pymoo | Multi-objective optimization (Pareto ranking) |

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific module tests
pytest tests/test_druglikeness.py -v

# Run with coverage
pytest tests/ --cov=core --cov=drug_selection --cov=lead_optimization --cov-report=html
```

---

## Roadmap

### Current (v0.1)
- [x] Project structure
- [x] Core computational modules
- [x] Drug Selection workflow
- [x] Lead Optimization workflow
- [x] CLI interface
- [ ] Report generation (JSON reports completed, HTML generator planned)
- [x] Test suite

### Planned (v0.2+)
- [ ] DeepChem / Chemprop ADMET backends
- [ ] ChemBERTa / MolFormer property prediction
- [ ] Molecular Dynamics refinement (OpenMM)
- [ ] MM/GBSA binding free energy rescoring
- [ ] Pharmacophore modeling
- [ ] QSAR model training
- [ ] Batch processing for large libraries

---

## License

MIT License

---

## Citation

If you use DrugLab in your research, please cite:

```
DrugLab: A Lightweight Command-Line Platform for Computational Drug Discovery
```
