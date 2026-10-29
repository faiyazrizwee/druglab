"""
DrugLab DAG-Based Workflow Engine.

Provides a dependency-graph-based execution engine that:
    1. Maintains a registry of module nodes with inputs, outputs, and dependencies.
    2. Given a set of selected modules, computes the transitive closure of dependencies.
    3. Topologically sorts the execution order.
    4. Determines which inputs need to be prompted.
    5. Executes modules in order, passing outputs as inputs to downstream modules.
    6. Collects all results in a structured results/ directory.

Design: modules are registered as ``ModuleNode`` objects. The engine resolves
dependencies automatically so that future modules (MD, MM/GBSA, Chemprop, …)
can be inserted by adding a single registration call.
"""

import os
import sys
import time
import logging
from dataclasses import dataclass, field
from collections import deque
from typing import Callable, Any

from utils.logger import (
    setup_logger, log_step, log_summary, console,
    set_workflow_log_dir, print_log_contents, get_all_log_files
)
from utils.config import load_config, save_config_snapshot, DrugLabConfig
from utils.io import ensure_dir

logger = setup_logger("workflow_engine", level="INFO")


# ──────────────────────────────────────────────────────────────
# Module Node Definition
# ──────────────────────────────────────────────────────────────

@dataclass
class ModuleNode:
    """
    Represents a single workflow module in the dependency graph.

    Attributes:
        name: Internal module identifier (e.g. "protein_preparation").
        display_name: Human-readable name for menus.
        menu_id: Integer ID shown in the interactive menu (0 = hidden).
        dependencies: List of module names this module depends on.
        inputs: List of context keys this module consumes.
        outputs: List of context keys this module produces.
        execute_fn: The function to call. Receives (ctx, config) → result dict.
        input_prompts: Map of input_key → prompt string for interactive input.
    """
    name: str
    display_name: str
    menu_id: int = 0
    dependencies: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    execute_fn: Callable | None = None
    input_prompts: dict[str, str] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────
# Manual Mode Input Definitions
#
# Each module defines what user-provided files/values are
# required when running in Manual Mode (i.e. without depending
# on upstream pipeline outputs).
#
# Format:
#   input_key: {
#       "prompt":       str  — question shown to user,
#       "check":        str  — "file" | "path" | "file_or_dir" | "file_or_blank" | "special_docking_pocket"
#       "optional":     bool — if True, can be left blank,
#       "context_key":  str  — key name to store value in context dict,
#       "upstream_keys": list — context keys that satisfy this input if present (from upstream module outputs),
#   }
#
# When gather_manual_inputs runs, for each input it checks:
#   1. Is context_key already in context? → skip
#   2. Is any upstream_key already in context? → skip (will be produced by an upstream module)
#   3. Otherwise → prompt the user
# ──────────────────────────────────────────────────────────────

MODULE_MANUAL_INPUTS: dict[str, dict[str, dict]] = {
    "protein_preparation": {
        "protein_pdb": {
            "prompt": "Protein PDB file path: ",
            "check": "file",
            "optional": False,
            "context_key": "protein_pdb",
            "upstream_keys": [],
        },
    },
    "ligand_preparation": {
        "ligand_library": {
            "prompt": "Ligand library (SDF file or directory): ",
            "check": "file_or_dir",
            "optional": False,
            "context_key": "ligand_library",
            "upstream_keys": [],
        },
    },
    "pocket_detection": {
        "protein_pdb": {
            "prompt": "Protein PDB file path: ",
            "check": "file",
            "optional": False,
            "context_key": "protein_pdb",
            # Satisfied if protein_preparation already ran (produces clean_pdb)
            "upstream_keys": ["clean_pdb"],
        },
    },
    "druglikeness": {
        "ligand_file": {
            "prompt": "Ligand file or directory (SDF/SMILES/CSV): ",
            "check": "file_or_dir",
            "optional": False,
            "context_key": "combined_sdf",
            # Satisfied if ligand_preparation already ran (produces combined_sdf)
            "upstream_keys": ["combined_sdf"],
        },
    },
    "admet": {
        "ligand_file": {
            "prompt": "Ligand file or directory for ADMET screening (SDF): ",
            "check": "file_or_dir",
            "optional": False,
            "context_key": "admet_input_sdf",
            # Satisfied if druglikeness already ran (produces passed_sdf)
            "upstream_keys": ["passed_sdf"],
        },
    },
    "docking": {
        "protein_pdbqt": {
            "prompt": "Prepared protein PDBQT file: ",
            "check": "file",
            "optional": False,
            "context_key": "pdbqt",
            "upstream_keys": ["pdbqt", "clean_pdb"],
            "fallback_paths": ["{output_dir}/protein_preparation/protein.pdbqt"]
        },
        "ligand_file": {
            "prompt": "Prepared ligand file (SDF or PDBQT): ",
            "check": "file_or_dir",
            "optional": False,
            "context_key": "docking_ligands",
            "upstream_keys": ["passed_sdf", "combined_sdf"],
            "fallback_paths": [
                "{output_dir}/admet/passed_ligands.sdf",
                "{output_dir}/druglikeness/passed_ligands.sdf",
                "{output_dir}/ligand_preparation/prepared_ligands.sdf"
            ]
        },
        "pocket_source": {
            "prompt": "",
            "check": "special_docking_pocket",
            "optional": False,
            "context_key": "_pocket_source_choice",
            "upstream_keys": ["docking_box"],
            "fallback_paths": ["{output_dir}/pocket_detection/docking_box.json"]
        },
    },
    "lead_selection": {
        "lead_selection_inputs": {
            "prompt": "",
            "check": "special_lead_selection",
            "optional": False,
            "context_key": "_lead_selection_inputs_collected",
            "upstream_keys": ["results_csv"],
            "fallback_paths": ["{output_dir}/docking/docking_results.csv"]
        },
    },
    "lead_optimization": {
        "lead_molecule_file": {
            "prompt": "Lead Molecule File: ",
            "check": "file",
            "optional": False,
            "context_key": "lead_molecule_file",
            "upstream_keys": ["selected_lead", "best_lead_sdf"],
            "fallback_paths": ["{output_dir}/lead_selection/best_lead.sdf"]
        },
        "protein_file": {
            "prompt": "Protein File: ",
            "check": "file",
            "optional": False,
            "context_key": "protein_file",
            "upstream_keys": ["pdbqt", "clean_pdb", "protein_pdb"],
            "fallback_paths": ["{output_dir}/protein_preparation/protein.pdbqt"]
        },
    },
}

def resolve_module_input(ctx: dict, keys_to_check: list[str], fallback_paths: list[str] = None) -> str | None:
    """
    Helper function to dynamically resolve upstream dependencies.
    Checks:
    1. Workflow Context (direct keys)
    2. Outputs from selected upstream modules (via _result_* dicts)
    3. Existing generated files (fallback_paths)
    """
    # 1. Workflow Context
    for k in keys_to_check:
        if k in ctx and ctx[k]:
            return ctx[k]
            
    # 2. Outputs from selected upstream modules
    # Look through all _result_* dicts in context
    for k, v in ctx.items():
        if k.startswith("_result_") and isinstance(v, dict):
            for out_key in keys_to_check:
                if out_key in v and v[out_key]:
                    return v[out_key]
                    
    # 3. Existing generated files
    if fallback_paths:
        for p in fallback_paths:
            # Substitute output_dir if present
            output_dir = ctx.get("output_dir", "results")
            resolved_p = p.replace("{output_dir}", output_dir)
            if os.path.exists(resolved_p):
                return resolved_p
                
    return None



# ──────────────────────────────────────────────────────────────
# Workflow Engine
# ──────────────────────────────────────────────────────────────

class WorkflowEngine:
    """
    DAG-based workflow execution engine.

    Usage:
        engine = WorkflowEngine()
        engine.register_defaults()
        engine.run_interactive()     # Mode 1
        engine.run_modules([...])    # Mode 2/3
    """

    def __init__(self):
        self._modules: dict[str, ModuleNode] = {}
        self._menu_map: dict[int, str] = {}  # menu_id → module_name

    # ── Registration ──

    def register(self, node: ModuleNode) -> None:
        """Register a module node in the engine."""
        self._modules[node.name] = node
        if node.menu_id > 0:
            self._menu_map[node.menu_id] = node.name

    def get_module(self, name: str) -> ModuleNode | None:
        """Look up a module by name."""
        return self._modules.get(name)

    def get_module_by_menu_id(self, menu_id: int) -> ModuleNode | None:
        """Look up a module by its menu ID."""
        mod_name = self._menu_map.get(menu_id)
        if mod_name:
            return self._modules.get(mod_name)
        return None

    # ── Dependency Resolution ──

    def resolve_dependencies(self, selected_names: list[str]) -> list[str]:
        """
        Given a list of selected module names, compute the full set of
        modules needed (including transitive dependencies) and return them
        in topological execution order.
        """
        # Compute transitive closure
        needed = set()
        queue = deque(selected_names)
        while queue:
            name = queue.popleft()
            if name in needed:
                continue
            needed.add(name)
            node = self._modules.get(name)
            if node:
                for dep in node.dependencies:
                    if dep not in needed:
                        queue.append(dep)

        # Topological sort (Kahn's algorithm)
        return self._topological_sort(needed)

    def _topological_sort(self, module_names: set[str]) -> list[str]:
        """Topologically sort a subset of modules."""
        # Build in-degree map within the subset
        in_degree = {n: 0 for n in module_names}
        adj: dict[str, list[str]] = {n: [] for n in module_names}

        for name in module_names:
            node = self._modules.get(name)
            if node:
                for dep in node.dependencies:
                    if dep in module_names:
                        adj[dep].append(name)
                        in_degree[name] += 1

        # Process nodes with in_degree 0
        queue = deque(n for n in module_names if in_degree[n] == 0)
        order = []

        while queue:
            n = queue.popleft()
            order.append(n)
            for neighbor in adj.get(n, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(module_names):
            missing = module_names - set(order)
            raise ValueError(f"Circular dependency detected among: {missing}")

        return order

    # ── Input Resolution (Automatic mode) ──

    def get_required_inputs(self, execution_order: list[str]) -> dict[str, str]:
        """
        Determine which inputs need to be prompted from the user.

        Returns a dict of {input_key: prompt_text} for inputs that are NOT
        produced by any module in the execution order.
        """
        # Collect all outputs produced by modules in the execution order
        produced = set()
        for name in execution_order:
            node = self._modules.get(name)
            if node:
                produced.update(node.outputs)

        # Collect required inputs not produced by upstream modules
        required = {}
        for name in execution_order:
            node = self._modules.get(name)
            if node:
                for inp in node.inputs:
                    if inp not in produced and inp not in required:
                        prompt = node.input_prompts.get(inp, f"Enter {inp}: ")
                        required[inp] = prompt

        return required

    # ── Input Collection (Manual mode) ──

    def gather_manual_inputs(self, selected_names: list[str], output_dir: str = "results") -> dict | None:
        """
        Interactively collect all required inputs for manual mode.

        Uses a smart deduplication strategy:
          1. Track outputs produced by upstream modules in this selection as we iterate.
          2. For each module's manual inputs, skip any input whose context_key
             or upstream_keys are already satisfied (either produced by an
             upstream module in the selection or already collected earlier).
          3. Only prompt the user for truly missing inputs.

        Returns a populated context dict, or None if the user aborts.
        """
        context: dict = {"output_dir": output_dir}

        # Track outputs produced by upstream modules in this selection as we iterate
        produced_so_far: set[str] = set()

        # Track which modules will actually need user input
        prompted_any = False

        for module_name in selected_names:
            node = self._modules.get(module_name)
            if node is None:
                continue

            manual_inputs = MODULE_MANUAL_INPUTS.get(module_name, {})
            if not manual_inputs:
                produced_so_far.update(node.outputs)
                continue

            # Determine which inputs for this module actually need prompting
            inputs_needed = []
            for inp_key, spec in manual_inputs.items():
                ctx_key = spec.get("context_key", inp_key)
                upstream_keys = spec.get("upstream_keys", [])
                check = spec.get("check", None)
                fallback_paths = spec.get("fallback_paths", [])

                # 1. Workflow Context & 2. Outputs from selected upstream modules
                if ctx_key in context:
                    continue
                if any(uk in context or uk in produced_so_far for uk in upstream_keys):
                    continue
                    
                # 3. Existing generated files
                resolved_from_file = False
                for p in fallback_paths:
                    output_dir = context.get("output_dir", "results")
                    resolved_p = p.replace("{output_dir}", output_dir)
                    if os.path.exists(resolved_p):
                        context[ctx_key] = resolved_p
                        resolved_from_file = True
                        break
                if resolved_from_file:
                    continue

                # For special handlers, also check if their target is satisfied
                if check == "special_docking_pocket" and "docking_box" in context:
                    continue
                if check == "special_docking_pocket" and "docking_box" in produced_so_far:
                    continue

                inputs_needed.append((inp_key, spec))

            if not inputs_needed:
                produced_so_far.update(node.outputs)
                continue  # All inputs for this module are auto-satisfied

            # Show module header only if we need to prompt
            prompted_any = True
            print()
            print(f"─── {node.display_name} ───")

            # Collect each needed input
            result = self._collect_module_inputs(module_name, inputs_needed, context)
            if result is None:
                return None  # User aborted

            produced_so_far.update(node.outputs)

        if not prompted_any:
            print()
            print("  All inputs will be auto-propagated from upstream modules.")

        return context

    def _collect_module_inputs(
        self,
        module_name: str,
        inputs_needed: list[tuple[str, dict]],
        context: dict,
    ) -> dict | None:
        """
        Prompt the user for a list of (inp_key, spec) inputs for a module.
        Returns the updated context dict, or None if aborted.
        """
        for inp_key, spec in inputs_needed:
            ctx_key = spec.get("context_key", inp_key)
            check = spec.get("check", None)
            optional = spec.get("optional", False)
            prompt_text = spec.get("prompt", f"Enter {inp_key}: ")

            # Special handler for docking pocket definition
            if check == "special_docking_pocket":
                if "docking_box" in context:
                    continue  # Already collected
                result = self._collect_docking_pocket(context)
                if result is None:
                    return None
                continue

            # Special handler for lead selection
            if check == "special_lead_selection":
                if "results_csv" in context:
                    continue  # Already collected or auto-propagated
                result = self._collect_lead_selection_inputs(context)
                if result is None:
                    return None
                continue

            # Special handler for protein target (Download or Local Path)
            if ctx_key in ("protein_pdb", "protein_file"):
                if ctx_key in context:
                    continue
                result = self._collect_protein_input(context, ctx_key)
                if result is None:
                    return None
                continue

            # Special handler for ligand library (Download or Local Path)
            if ctx_key in ("ligand_library", "ligand_file"):
                if ctx_key in context:
                    continue
                result = self._collect_ligand_input(context, ctx_key)
                if result is None:
                    return None
                continue

            # Skip if already populated by a previous module's collection
            if ctx_key in context:
                continue

            while True:
                print(f"{prompt_text}", end="", flush=True)
                try:
                    value = input().strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nCancelled.")
                    return None

                if not value:
                    if optional:
                        break  # Leave optional inputs empty
                    print(f"  [!] This field is required. Please provide a value.")
                    continue

                # Normalize path
                value = os.path.normpath(value)

                # Validate based on check type
                if check == "file":
                    if not os.path.isfile(value):
                        print(f"  [!] File not found: {value}")
                        print(f"      Please check the path and try again.")
                        continue
                elif check == "path":
                    if not os.path.exists(value):
                        print(f"  [!] Path not found: {value}")
                        print(f"      Please check the path and try again.")
                        continue
                elif check == "file_or_dir":
                    if not os.path.isfile(value) and not os.path.isdir(value):
                        print(f"  [!] File or directory not found: {value}")
                        print(f"      Please check the path and try again.")
                        continue
                elif check == "file_or_blank":
                    if value and not os.path.isfile(value):
                        print(f"  [!] File not found: {value}")
                        print(f"      Please check the path and try again.")
                        continue

                context[ctx_key] = value
                break

        return context

    def _collect_protein_input(self, context: dict, ctx_key: str = "protein_pdb") -> str | None:
        """
        Interactively collect protein input: allow local path or download from RCSB PDB.
        """
        print()
        print("Provide protein target:")
        print("  [1] Enter local PDB file path")
        print("  [2] Download from RCSB PDB by ID (e.g. 7DTD, 4MZI, 6LU7)")
        print("Choice [1]: ", end="", flush=True)
        try:
            choice = input().strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return None

        if choice == "2":
            while True:
                print("Enter 4-character RCSB PDB ID: ", end="", flush=True)
                try:
                    pdb_id = input().strip().upper()
                except (EOFError, KeyboardInterrupt):
                    print("\nCancelled.")
                    return None
                if not pdb_id:
                    print("  [!] PDB ID is required.")
                    continue
                try:
                    from utils.fetch import fetch_pdb
                    dest_file = os.path.join("protein", f"{pdb_id}.pdb")
                    path = fetch_pdb(pdb_id, dest_file)
                    print(f"  ✓ Downloaded PDB structure to: {path}")
                    context[ctx_key] = path
                    return path
                except Exception as e:
                    print(f"  [!] Failed to download PDB {pdb_id}: {e}")
                    print("      Please try another ID or choose local file.")
                    return self._collect_protein_input(context, ctx_key)
        else:
            default_hint = ""
            if os.path.isfile("protein/7DTD.pdb"):
                default_hint = " [default: protein/7DTD.pdb]"
            while True:
                print(f"Protein PDB file path{default_hint}: ", end="", flush=True)
                try:
                    path = input().strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nCancelled.")
                    return None
                if not path and default_hint:
                    path = "protein/7DTD.pdb"
                if not path:
                    print("  [!] Path is required.")
                    continue

                path = os.path.normpath(path)
                if not os.path.isfile(path):
                    if len(path) == 4 and path.isalnum():
                        print(f"  Local file '{path}' not found. Download PDB '{path.upper()}' from RCSB? [Y/n]: ", end="", flush=True)
                        try:
                            dl_choice = input().strip().lower()
                        except (EOFError, KeyboardInterrupt):
                            dl_choice = "n"
                        if dl_choice not in ("n", "no"):
                            try:
                                from utils.fetch import fetch_pdb
                                dest_file = os.path.join("protein", f"{path.upper()}.pdb")
                                p = fetch_pdb(path.upper(), dest_file)
                                print(f"  ✓ Downloaded PDB structure to: {p}")
                                context[ctx_key] = p
                                return p
                            except Exception as dl_err:
                                print(f"  [!] Download failed: {dl_err}")
                    print(f"  [!] File not found: {path}. Please check path and try again.")
                    continue

                context[ctx_key] = path
                return path

    def _collect_ligand_input(self, context: dict, ctx_key: str = "ligand_library") -> str | None:
        """
        Interactively collect ligand input: allow local path or download from PubChem.
        """
        print()
        print("Provide ligand library:")
        print("  [1] Enter local SDF file or directory")
        print("  [2] Download compound from PubChem by name (e.g. apigenin, quercetin, aspirin)")
        print("  [3] Download compound from PubChem by CID")
        print("Choice [1]: ", end="", flush=True)
        try:
            choice = input().strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return None

        if choice == "2":
            while True:
                print("Enter compound name: ", end="", flush=True)
                try:
                    name = input().strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nCancelled.")
                    return None
                if not name:
                    print("  [!] Compound name is required.")
                    continue
                try:
                    from utils.fetch import fetch_ligand_by_name
                    safe_name = name.lower().replace(" ", "_")
                    dest_file = os.path.join("ligands", f"{safe_name}.sdf")
                    path = fetch_ligand_by_name(name, dest_file)
                    print(f"  ✓ Downloaded ligand to: {path}")
                    context[ctx_key] = path
                    return path
                except Exception as e:
                    print(f"  [!] Failed to download ligand '{name}': {e}")
                    return self._collect_ligand_input(context, ctx_key)

        elif choice == "3":
            while True:
                print("Enter PubChem CID: ", end="", flush=True)
                try:
                    cid = input().strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nCancelled.")
                    return None
                if not cid:
                    print("  [!] CID is required.")
                    continue
                try:
                    from utils.fetch import fetch_ligand_by_cid
                    dest_file = os.path.join("ligands", f"CID_{cid}.sdf")
                    path = fetch_ligand_by_cid(cid, dest_file)
                    print(f"  ✓ Downloaded ligand to: {path}")
                    context[ctx_key] = path
                    return path
                except Exception as e:
                    print(f"  [!] Failed to download CID '{cid}': {e}")
                    return self._collect_ligand_input(context, ctx_key)

        else:
            default_hint = ""
            if os.path.isdir("ligands"):
                default_hint = " [default: ligands/]"
            while True:
                print(f"Ligand file or directory path{default_hint}: ", end="", flush=True)
                try:
                    path = input().strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nCancelled.")
                    return None
                if not path and default_hint:
                    path = "ligands"
                if not path:
                    print("  [!] Path is required.")
                    continue

                path = os.path.normpath(path)
                if not os.path.isfile(path) and not os.path.isdir(path):
                    print(f"  Local path '{path}' not found. Download '{path}' from PubChem? [Y/n]: ", end="", flush=True)
                    try:
                        dl_choice = input().strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        dl_choice = "n"
                    if dl_choice not in ("n", "no"):
                        try:
                            from utils.fetch import fetch_ligand_by_name
                            safe_name = path.lower().replace(" ", "_")
                            dest_file = os.path.join("ligands", f"{safe_name}.sdf")
                            p = fetch_ligand_by_name(path, dest_file)
                            print(f"  ✓ Downloaded ligand to: {p}")
                            context[ctx_key] = p
                            return p
                        except Exception as dl_err:
                            print(f"  [!] Download failed: {dl_err}")
                    print(f"  [!] File or directory not found: {path}. Please try again.")
                    continue

                context[ctx_key] = path
                return path

    def _collect_docking_pocket(self, context: dict) -> dict | None:
        """
        Interactively collect the pocket/docking-box definition.
        Asks whether the user has a JSON file or wants to enter coords manually.
        """
        print()
        print("Do you already have a pocket definition JSON file?")
        print("  [Y] Yes — provide file path")
        print("  [N] No  — enter docking box coordinates manually")
        print("Choice [Y/n]: ", end="", flush=True)
        try:
            choice = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return None

        if choice in ("n", "no"):
            # Collect manual box coordinates
            print()
            print("Enter docking box coordinates:")
            coords: dict = {}
            fields = [
                ("center_x", "  Center X: "),
                ("center_y", "  Center Y: "),
                ("center_z", "  Center Z: "),
                ("size_x",   "  Size X:   "),
                ("size_y",   "  Size Y:   "),
                ("size_z",   "  Size Z:   "),
            ]
            for field_key, prompt in fields:
                while True:
                    print(prompt, end="", flush=True)
                    try:
                        raw = input().strip()
                    except (EOFError, KeyboardInterrupt):
                        print("\nCancelled.")
                        return None
                    try:
                        coords[field_key] = float(raw)
                        break
                    except ValueError:
                        print(f"  [!] Please enter a valid number.")

            context["docking_box"] = coords
            context["_docking_box_type"] = "coords"
        else:
            # Collect JSON file path
            while True:
                print("Pocket JSON file path: ", end="", flush=True)
                try:
                    path = input().strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nCancelled.")
                    return None
                if not path:
                    print("  [!] Path is required.")
                    continue
                if not os.path.isfile(path):
                    print(f"  [!] File not found: {path}")
                    continue
                context["docking_box"] = path
                context["_docking_box_type"] = "file"
                break

        return context

    def _collect_lead_selection_inputs(self, context: dict) -> dict | None:
        """Interactively collect inputs for standalone lead selection."""
        print()
        print("Do you already have docking results?")
        print("  [Y/N]: ", end="", flush=True)
        try:
            ans = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return None

        if ans not in ("y", "yes"):
            print("  [!] Docking results are required for Lead Selection. Exiting.")
            return None

        while True:
            print("Docking Results CSV: ", end="", flush=True)
            try:
                path = input().strip()
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled.")
                return None
            path = os.path.normpath(path)
            if not os.path.isfile(path):
                print(f"  [!] File not found: {path}")
                continue
            context["results_csv"] = path
            break

        print()
        print("Do you have ADMET results?")
        print("  [Y/N]: ", end="", flush=True)
        try:
            ans = input().strip().lower()
            if ans in ("y", "yes"):
                while True:
                    print("ADMET Results CSV: ", end="", flush=True)
                    path = input().strip()
                    path = os.path.normpath(path)
                    if not os.path.isfile(path):
                        print(f"  [!] File not found: {path}")
                        continue
                    context["admet_csv"] = path
                    break
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return None

        print()
        print("Do you have Drug-Likeness results?")
        print("  [Y/N]: ", end="", flush=True)
        try:
            ans = input().strip().lower()
            if ans in ("y", "yes"):
                while True:
                    print("Drug-Likeness Report or CSV: ", end="", flush=True)
                    path = input().strip()
                    path = os.path.normpath(path)
                    if not os.path.isfile(path):
                        print(f"  [!] File not found: {path}")
                        continue
                    context["descriptors_csv"] = path
                    break
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return None

        context["_lead_selection_inputs_collected"] = True
        return context

    def _collect_lead_optimization_inputs(self, context: dict) -> dict | None:
        """Interactively collect inputs for standalone lead optimization."""
        print()
        print("Do you already have a lead molecule?")
        print("  [Y/N]: ", end="", flush=True)
        try:
            ans = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return None

        if ans not in ("y", "yes"):
            print("  [!] A lead molecule is required for Optimization. Exiting.")
            return None

        while True:
            print("Lead Molecule SDF: ", end="", flush=True)
            try:
                path = input().strip()
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled.")
                return None
            path = os.path.normpath(path)
            if not os.path.isfile(path):
                print(f"  [!] File not found: {path}")
                continue
            context["lead_molecule"] = path
            break

        return context

    # ── Execution ──

    def run_modules(
        self,
        selected_names: list[str],
        context: dict | None = None,
        config: DrugLabConfig | None = None,
        output_dir: str = "results",
        skip_dependencies: bool = False,
    ) -> dict:
        """
        Execute a set of modules with automatic dependency resolution.

        Args:
            selected_names: List of module names to execute.
            context: Pre-populated context dict (input paths, etc.).
            config: DrugLabConfig instance.
            output_dir: Output directory.
            skip_dependencies: If True, execute selected_names as-is without
                               re-resolving dependencies (used in manual mode).

        Returns:
            Dict of module_name → result dict.
        """
        if config is None:
            config = load_config()

        if context is None:
            context = {}

        context["output_dir"] = output_dir
        ensure_dir(output_dir)
        save_config_snapshot(config, output_dir)
        set_workflow_log_dir(os.path.join(output_dir, "logs"))

        # Resolve dependencies and sort unless skipping
        if skip_dependencies:
            execution_order = selected_names
        else:
            execution_order = self.resolve_dependencies(selected_names)
        
        console.print(f"[bold]Execution plan:[/bold] [cyan]{' → '.join(execution_order)}[/cyan]\n")

        results = {}
        for name in execution_order:
            node = self._modules.get(name)
            if node is None:
                logger.warning(f"Module '{name}' not registered; skipping.")
                continue

            if node.execute_fn is None:
                logger.warning(f"Module '{name}' has no execute function; skipping.")
                continue

            try:
                result = node.execute_fn(context, config)
                results[name] = result

                if isinstance(result, dict) and result.get("status") == "aborted":
                    console.print(f"[yellow]Module '{node.display_name}' aborted. Halting workflow gracefully.[/yellow]")
                    break

                # Update context with outputs
                if isinstance(result, dict):
                    for key in node.outputs:
                        if key in result:
                            context[key] = result[key]
                    # Also store the full result for downstream access
                    context[f"_result_{name}"] = result

            except Exception as e:
                logger.error(f"Module '{name}' failed: {e}")
                results[name] = {"status": "failed", "error": str(e)}
                raise

        # Print summary
        summary = {}
        for name, result in results.items():
            node = self._modules.get(name)
            display = node.display_name if node else name
            summary[display] = {
                "status": result.get("status", "unknown"),
                "duration": result.get("duration", "—"),
            }
        log_summary(logger, summary)

        # ── Figure Generation: Summary ──
        try:
            from reports.generator import generate_summary_figures
            counts = {}
            if "ligand_preparation" in results:
                # Try getting total_input, fallback to total_ligands, or total_processed
                inp = results["ligand_preparation"].get("total_input")
                if inp is None:
                    inp = results["ligand_preparation"].get("total_ligands", 0)
                counts["Input Ligands"] = inp
            if "druglikeness" in results:
                counts["Drug-Like"] = results["druglikeness"].get("total_passed", 0)
            if "admet" in results:
                counts["ADMET Passed"] = results["admet"].get("total_passed", 0)
            if "docking" in results:
                counts["Docked"] = results["docking"].get("completed", 0)
            if "lead_selection" in results:
                counts["Selected Leads"] = results["lead_selection"].get("total_selected", 0)
            if "lead_optimization" in results:
                if "ranking" in results["lead_optimization"]:
                    counts["Optimized Analogues"] = results["lead_optimization"]["ranking"].get("total_ranked", 0)

            if len(counts) >= 2:
                gen_figs = generate_summary_figures(counts, output_dir)
                if gen_figs:
                    print(f"\nGenerated {len(gen_figs)} summary figures:")
                    print(f"{os.path.join(output_dir, 'figures', 'summary')}/")
                    for f in gen_figs:
                        print(f"  ✓ {os.path.basename(f)}")
                    print()
        except Exception as e:
            logger.error(f"Failed to generate summary figures: {e}")

        # Auto-generate interactive HTML report at completion
        try:
            from reports.generator import generate_report
            rep_path = generate_report(output_dir=output_dir)
            abs_rep = Path(rep_path).resolve().as_uri()
            console.print(f"[bold green]✓ Interactive HTML Report generated:[/bold green] [cyan]{rep_path}[/cyan]")
            console.print(f"  [dim]Open in browser (Ctrl+click):[/dim] [link={abs_rep}]{rep_path}[/link]\n")
        except Exception as rep_err:
            logger.debug(f"Could not auto-generate report: {rep_err}")

        logs_dir = os.path.join(output_dir, "logs")
        console.print(f"[dim]💡 Tip: Logs for each module are minimized above and saved in [cyan]{logs_dir}[/cyan][/dim]")
        console.print("[dim]   Ctrl+click any log link above to open it, or run 'druglab report' to view them in the browser report.[/dim]\n")

        return results

    def run_interactive(self, config: DrugLabConfig | None = None) -> dict | None:
        """
        Run the interactive workflow menu with Rich styling, presets, and utilities.

        Returns results dict or None if user exits.
        """
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()

        banner_text = (
            "[bold cyan]🧬 DrugLab[/bold cyan] — [white]Computational Drug Discovery Platform[/white]\n"
            "[dim]A lightweight platform for virtual screening, docking, ADMET & lead optimization[/dim]"
        )
        console.print()
        console.print(Panel(banner_text, border_style="cyan", padding=(0, 2)))

        table = Table(title="Select Workflow or Computational Modules", border_style="dim", show_header=True, header_style="bold cyan")
        table.add_column("Option", style="bold yellow", width=10, justify="center")
        table.add_column("Category", style="bold magenta", width=12)
        table.add_column("Workflow / Module Name", style="bold white", width=28)
        table.add_column("Description", style="dim", width=46)

        # Presets
        table.add_row("[A]", "Pipeline", "Complete Virtual Screening", "Steps 1–7: Protein Prep → ADMET → Docking → Leads")
        table.add_row("[B]", "Pipeline", "Lead Optimization", "Step 8: Bioisostere, scaffold hop, SAScore & ranking")
        table.add_row("[C]", "Custom", "Custom Module Range", "Run selected step range (e.g. 1-4, or 1,2,6)")
        table.add_section()

        # Modules
        module_descriptions = {
            1: "Clean PDB, add hydrogens, repair atoms & PDBQT",
            2: "Conformer generation, energy minimization & PDBQT",
            3: "Geometric cavity detection & AutoDock Vina box",
            4: "Lipinski, Veber, Ghose, PAINS structural alerts",
            5: "ADMET-AI predictions (BBB, toxicity, metabolism)",
            6: "AutoDock Vina binding affinity calculation",
            7: "Multi-criteria ranking with LE & LipE metrics",
            8: "Analogue generation, SAScore filtering & optimization",
        }

        for i in range(1, 9):
            node = self.get_module_by_menu_id(i)
            if node:
                desc = module_descriptions.get(i, "")
                table.add_row(f"[{i}]", "Module", node.display_name, desc)

        table.add_section()
        table.add_row("[R]", "Utility", "Generate HTML Report", "Create standalone interactive report from output/")
        table.add_row("[F]", "Utility", "Fetch PDB / PubChem", "Download PDB from RCSB or ligand from PubChem")
        table.add_row("[Q]", "System", "Exit", "Quit DrugLab")

        console.print(table)
        console.print()

        try:
            selection_raw = input("Selection [default: A (Complete Screening)]: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Exiting DrugLab.[/dim]")
            return None

        choice = selection_raw.upper() if selection_raw else "A"

        # Handle Utilities & Exits
        if choice in ("Q", "9", "EXIT", "QUIT"):
            console.print("[dim]Exiting DrugLab. Happy discovery![/dim]")
            return None

        if choice in ("R", "REPORT"):
            print()
            try:
                rep_dir = input("Enter output directory with results [output]: ").strip() or "output"
            except (EOFError, KeyboardInterrupt):
                return None
            try:
                from reports.generator import generate_report
                path = generate_report(output_dir=rep_dir)
                console.print(f"\n[bold green]✓ Standalone HTML report generated successfully:[/bold green] [underline]{path}[/underline]\n")
            except Exception as rep_err:
                console.print(f"[bold red]Failed to generate report:[/bold red] {rep_err}")
            return None

        if choice in ("F", "FETCH"):
            print()
            print("What would you like to fetch?")
            print("  [1] Download PDB structure from RCSB PDB")
            print("  [2] Download chemical ligand SDF from PubChem by name")
            print("  [3] Download chemical ligand SDF from PubChem by CID")
            try:
                f_type = input("Choice [1]: ").strip() or "1"
            except (EOFError, KeyboardInterrupt):
                return None
            if f_type == "2":
                name = input("Enter compound name (e.g. apigenin): ").strip()
                if name:
                    from utils.fetch import fetch_ligand_by_name
                    p = fetch_ligand_by_name(name)
                    console.print(f"[bold green]✓ Downloaded ligand to:[/bold green] {p}")
            elif f_type == "3":
                cid = input("Enter PubChem CID: ").strip()
                if cid:
                    from utils.fetch import fetch_ligand_by_cid
                    p = fetch_ligand_by_cid(cid)
                    console.print(f"[bold green]✓ Downloaded ligand to:[/bold green] {p}")
            else:
                pdb = input("Enter 4-character PDB ID (e.g. 7DTD): ").strip().upper()
                if pdb:
                    from utils.fetch import fetch_pdb
                    p = fetch_pdb(pdb)
                    console.print(f"[bold green]✓ Downloaded PDB structure to:[/bold green] {p}")
            return None

        # Determine step numbers
        if choice in ("A", "ALL", "FULL"):
            selections = [1, 2, 3, 4, 5, 6, 7]
            console.print("[cyan]Selected:[/cyan] [bold]Complete Virtual Screening Pipeline (Steps 1 → 7)[/bold]")
        elif choice in ("B", "OPT", "OPTIMIZE"):
            selections = [8]
            console.print("[cyan]Selected:[/cyan] [bold]Lead Optimization Pipeline (Step 8)[/bold]")
        elif choice in ("C", "CUSTOM"):
            try:
                custom_str = input("Enter steps to run (e.g. 1-4, or 1,2,6): ").strip()
                selections = self._parse_step_selection(custom_str)
            except (EOFError, KeyboardInterrupt):
                return None
        else:
            selections = self._parse_step_selection(selection_raw)

        if not selections:
            console.print("[yellow]No valid workflow steps selected.[/yellow]")
            return None

        selected_names = []
        for sel_id in selections:
            node = self.get_module_by_menu_id(sel_id)
            if node:
                selected_names.append(node.name)
            else:
                console.print(f"[yellow]Unknown option: {sel_id}[/yellow]")

        if not selected_names:
            console.print("[yellow]No valid modules selected.[/yellow]")
            return None

        if config is None:
            config = load_config()

        return self._run_workflow(selected_names, config, console)

    @staticmethod
    def _parse_step_selection(selection_str: str) -> list[int]:
        """Parse comma/space separated numbers and ranges (e.g. '1-4', '1, 2, 6')."""
        steps = set()
        parts = selection_str.replace(",", " ").split()
        for part in parts:
            if "-" in part:
                sub = part.split("-")
                if len(sub) == 2 and sub[0].isdigit() and sub[1].isdigit():
                    start, end = int(sub[0]), int(sub[1])
                    for x in range(start, end + 1):
                        steps.add(x)
            elif ".." in part:
                sub = part.split("..")
                if len(sub) == 2 and sub[0].isdigit() and sub[1].isdigit():
                    start, end = int(sub[0]), int(sub[1])
                    for x in range(start, end + 1):
                        steps.add(x)
            elif part.isdigit():
                steps.add(int(part))
        return sorted(list(steps))

    def _run_workflow(
        self,
        selected_names: list[str],
        config: DrugLabConfig,
        console: Any
    ) -> dict | None:
        """Run the workflow path (executes chosen steps strictly)."""
        def get_menu_id(name):
            node = self._modules.get(name)
            return node.menu_id if node else 999
        execution_order = sorted(selected_names, key=get_menu_id)

        context: dict = {
            "interactive": True
        }

        # Output directory
        print()
        try:
            out = input("Output directory [results]: ").strip()
            output_dir = out if out else "results"
        except (EOFError, KeyboardInterrupt):
            output_dir = "results"
            
        context["output_dir"] = output_dir

        # Gather inputs — smart deduplication skips inputs satisfied by upstream modules
        result = self.gather_manual_inputs(execution_order, output_dir)
        if result is None:
            return None  # User aborted or lacked required files
        context.update(result)

        # Compute auto-propagated relationships for display
        produced_keys: set[str] = set()
        auto_propagated: list[str] = []
        for name in execution_order:
            node = self._modules.get(name)
            if node is None:
                continue
            manual_inputs = MODULE_MANUAL_INPUTS.get(name, {})
            for inp_key, spec in manual_inputs.items():
                ctx_key = spec.get("context_key", inp_key)
                upstream_keys = spec.get("upstream_keys", [])
                # If the input was satisfied by upstream and not user-provided
                if ctx_key not in result and any(uk in produced_keys for uk in upstream_keys):
                    # Find which upstream module produces it
                    for upstream_name in execution_order:
                        upstream_node = self._modules.get(upstream_name)
                        if upstream_node and any(uk in upstream_node.outputs for uk in upstream_keys):
                            auto_propagated.append(
                                f"{node.display_name} ← uses output from {upstream_node.display_name}"
                            )
                            break
            if node:
                produced_keys.update(node.outputs)

        # Display Summary
        console.print()
        console.print("=" * 60)
        console.print("[bold cyan]DrugLab Execution Plan[/bold cyan]")
        console.print("=" * 60)
        console.print("[bold]Steps to execute:[/bold]")
        for i, name in enumerate(execution_order):
            node = self._modules[name]
            console.print(f"  {i+1}. {node.display_name}")
        console.print()

        # Show user-provided inputs
        user_inputs = {k: v for k, v in context.items()
                       if k not in ("interactive", "workflow_mode")}
        if user_inputs:
            console.print("[bold]User-provided inputs:[/bold]")
            for k, v in user_inputs.items():
                if isinstance(v, dict):
                    console.print(f"  - {k}: [yellow]{v}[/yellow]")
                else:
                    console.print(f"  - {k}: [yellow]{v}[/yellow]")
            console.print()

        # Show auto-propagated connections
        if auto_propagated:
            console.print("[bold]Auto-propagated between stages:[/bold]")
            for desc in auto_propagated:
                console.print(f"  [dim]→ {desc}[/dim]")
            console.print()

        console.print(f"  Output directory: [yellow]{output_dir}[/yellow]")
        console.print()
        console.print("=" * 60)
        console.print()

        try:
            proceed = input("Proceed with execution? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Cancelled.[/dim]")
            return None

        if proceed not in ("", "y", "yes"):
            console.print("[yellow]Execution cancelled by user.[/yellow]")
            return None

        console.print()
        console.print("[bold green]Starting workflow...[/bold green]")
        console.print()

        # Set pocket standalone flag — pocket detection shows the full interactive
        # selection menu when standalone, vs. the simpler confirmation prompt in pipelines.
        downstream_of_pocket = {"docking", "lead_selection", "lead_optimization"}
        has_downstream = bool(set(execution_order) & downstream_of_pocket)
        context["_pocket_standalone"] = (
            "pocket_detection" in execution_order and not has_downstream
        )

        return self.run_modules(
            selected_names=execution_order,
            context=context,
            config=config,
            output_dir=output_dir,
            skip_dependencies=True,
        )


# ──────────────────────────────────────────────────────────────
# Default Module Registrations
# ──────────────────────────────────────────────────────────────

def _exec_protein_preparation(ctx: dict, config: DrugLabConfig) -> dict:
    """Execute protein preparation module."""
    from core.protein_preparation import prepare_protein

    input_pdb = ctx.get("protein_pdb")
    if not input_pdb:
        raise ValueError("protein_pdb not provided in context.")

    return prepare_protein(
        input_pdb=input_pdb,
        output_dir=ctx["output_dir"],
        chain_id=config.protein.chain_id,
        remove_water=config.protein.remove_water,
        remove_heterogens=config.protein.remove_heterogens,
        add_hydrogens=config.protein.add_hydrogens,
        fix_missing_residues=config.protein.fix_missing_residues,
        fix_missing_atoms=config.protein.fix_missing_atoms,
        ph=config.protein.ph,
        keep_ligands=config.protein.keep_ligands,
    )


def _exec_ligand_preparation(ctx: dict, config: DrugLabConfig) -> dict:
    """Execute ligand preparation module."""
    from core.ligand_preparation import prepare_ligands

    input_path = ctx.get("ligand_library")
    if not input_path:
        raise ValueError("ligand_library not provided in context.")

    return prepare_ligands(
        input_path=input_path,
        output_dir=ctx["output_dir"],
        add_hydrogens=config.ligand.add_hydrogens,
        generate_3d=config.ligand.generate_3d,
        num_conformers=config.ligand.num_conformers,
        energy_minimize=config.ligand.energy_minimize,
        ph=config.ligand.ph,
        max_tautomers=config.ligand.max_tautomers,
        convert_to_pdbqt=config.ligand.convert_to_pdbqt,
    )


def _exec_pocket_detection(ctx: dict, config: DrugLabConfig) -> dict:
    """Execute pocket detection module."""
    from core.pocket_detection import (
        detect_pockets,
        interactive_pocket_selection,
        prompt_pocket_confirmation_with_selection,
        display_selected_pocket_summary,
    )
    import gemmi

    # Unified resolution: prefer cleaned PDB from upstream, fall back to raw input
    protein_pdb = resolve_module_input(ctx, ["clean_pdb", "protein_pdb"])

    if not protein_pdb:
        raise ValueError("No protein PDB available for pocket detection.")

    kwargs = {
        "protein_pdb": protein_pdb,
        "output_dir": ctx["output_dir"],
        "method": config.pocket.method,
        "selection_mode": config.pocket.selection_mode,
        "min_pocket_size": config.pocket.min_pocket_size,
        "max_pockets": config.pocket.max_pockets,
        "padding": config.pocket.padding,
        "max_box_dimension": config.pocket.max_box_dimension,
        "warn_large_boxes": config.pocket.warn_large_boxes,
        "interactive": False,
    }
    if config.pocket.reference_ligand:
        kwargs["reference_ligand"] = config.pocket.reference_ligand
    if config.pocket.manual_center:
        kwargs["manual_center"] = tuple(config.pocket.manual_center)
    if config.pocket.manual_size:
        kwargs["manual_size"] = tuple(config.pocket.manual_size)

    # Run detection
    res = detect_pockets(**kwargs)
    pockets = res.get("pockets", [])

    # Interactive pocket selection (only when running interactively)
    if ctx.get("interactive"):
        # Load structure for manual pocket residue extraction
        doc = gemmi.read_structure(protein_pdb)
        max_dim = config.pocket.max_box_dimension

        # Determine if pocket detection is standalone or part of a larger pipeline
        # Standalone = only pocket_detection in the execution plan
        # Pipeline = pocket_detection + other downstream modules (e.g. docking)
        is_standalone = ctx.get("_pocket_standalone", False)

        if is_standalone:
            # Full interactive selection menu
            selected, method = interactive_pocket_selection(
                pockets, max_dim, doc, protein_pdb=protein_pdb
            )
        else:
            # Pipeline: confirmation prompt + fallback to selection menu
            selected, method = prompt_pocket_confirmation_with_selection(
                pockets, max_dim, doc, protein_pdb=protein_pdb
            )

        # Update result with user's selection
        res["selected_pocket"] = selected
        res["selection_method"] = method
        res["docking_box"] = {
            "center": selected["center"],
            "size": selected["size"],
        }

        # Persist selected_pocket.json
        import os
        from utils.io import write_json
        pocket_dir = os.path.join(ctx["output_dir"], "pocket_detection")
        write_json(selected, os.path.join(pocket_dir, "selected_pocket.json"))
        write_json(res["docking_box"], os.path.join(pocket_dir, "docking_box.json"))

        # Write manual_pocket.json if manual entry or graphical GUI was used
        if method in ("manual", "graphical_gui"):
            write_json(selected, os.path.join(pocket_dir, "manual_pocket.json"))

    # Store in context
    ctx["_selected_pocket"] = res.get("selected_pocket", res.get("best_pocket", {}))
    ctx["_selection_method"] = res.get("selection_method", "best_score")

    return res


def _exec_druglikeness(ctx: dict, config: DrugLabConfig) -> dict:
    """Execute drug-likeness screening module."""
    from core.druglikeness import screen_druglikeness

    # Unified resolution: check direct context key, then upstream output, then fallback
    input_sdf = resolve_module_input(ctx, ["combined_sdf", "prepared_ligands_sdf"], [
        f"{ctx.get('output_dir', 'results')}/ligand_preparation/prepared_ligands.sdf"
    ])

    if not input_sdf:
        raise ValueError(
            "No ligand SDF found for Drug-Likeness Screening.\n"
            "Please provide the ligand file when prompted."
        )

    profile = config.druglikeness.profile
    evaluation_mode = config.druglikeness.evaluation_mode
    minimum_rule_pass_fraction = config.druglikeness.minimum_rule_pass_fraction
    brenk_mode = config.druglikeness.brenk_mode
    rules = config.druglikeness.rules

    # Interactive prompting once per session
    if ctx.get("interactive") and "_druglikeness_profile" not in ctx:
        from core.druglikeness import prompt_profile_selection
        prof, resolved_rules = prompt_profile_selection()
        ctx["_druglikeness_profile"] = prof
        ctx["_druglikeness_rules"] = resolved_rules
        ctx["_druglikeness_eval_mode"] = "majority"

    if "_druglikeness_profile" in ctx:
        profile = ctx["_druglikeness_profile"]
        rules = ctx["_druglikeness_rules"]
        evaluation_mode = ctx.get("_druglikeness_eval_mode", "majority")

    return screen_druglikeness(
        input_sdf=input_sdf,
        output_dir=ctx["output_dir"],
        profile=profile,
        rules=rules,
        evaluation_mode=evaluation_mode,
        minimum_rule_pass_fraction=minimum_rule_pass_fraction,
        brenk_mode=brenk_mode,
        custom_filters=config.druglikeness.custom_filters,
        max_violations=config.druglikeness.max_violations,
    )


def _exec_admet(ctx: dict, config: DrugLabConfig) -> dict:
    """Execute ADMET screening module."""
    from core.admet import screen_admet

    # Unified resolution: manual input key → upstream druglikeness output → fallback
    input_sdf = resolve_module_input(ctx, ["admet_input_sdf", "passed_sdf", "combined_sdf"], [
        f"{ctx.get('output_dir', 'results')}/druglikeness/passed_ligands.sdf",
        f"{ctx.get('output_dir', 'results')}/ligand_preparation/prepared_ligands.sdf"
    ])

    if not input_sdf:
        raise ValueError(
            "No ligand SDF found for ADMET Screening.\n"
            "Please provide the ligand file when prompted."
        )

    profile = config.admet.profile
    target_type = config.admet.target_type

    # Interactive prompting once per session
    if ctx.get("interactive") and "_admet_profile" not in ctx:
        from core.admet import prompt_admet_profile_selection
        prof = prompt_admet_profile_selection()
        ctx["_admet_profile"] = prof

    if "_admet_profile" in ctx:
        profile = ctx["_admet_profile"]

    return screen_admet(
        input_sdf=input_sdf,
        output_dir=ctx["output_dir"],
        profile=profile,
        target_type=target_type,
        filter_toxic=config.admet.filter_toxic,
        filter_low_absorption=config.admet.filter_low_absorption,
        thresholds=config.admet.thresholds,
    )


def _exec_docking(ctx: dict, config: DrugLabConfig) -> dict:
    """Execute molecular docking module."""
    from core.docking import run_docking, prompt_exhaustiveness_selection
    from core.pocket_detection import display_selected_pocket_summary

    # Unified resolution for protein PDBQT
    protein_pdbqt = resolve_module_input(ctx, ["pdbqt", "protein_pdbqt"], [
        f"{ctx.get('output_dir', 'results')}/protein_preparation/protein.pdbqt"
    ])

    # Unified resolution for ligands
    ligands = resolve_module_input(ctx, ["docking_ligands", "passed_sdf", "combined_sdf"], [
        f"{ctx.get('output_dir', 'results')}/admet/passed_ligands.sdf",
        f"{ctx.get('output_dir', 'results')}/druglikeness/passed_ligands.sdf",
        f"{ctx.get('output_dir', 'results')}/ligand_preparation/prepared_ligands.sdf"
    ])

    # Unified resolution for docking box (may be file path or coords dict)
    docking_box = resolve_module_input(ctx, ["docking_box"], [
        f"{ctx.get('output_dir', 'results')}/pocket_detection/docking_box.json"
    ])

    if not protein_pdbqt:
        raise ValueError(
            "No protein PDBQT found for Docking.\n"
            "Please provide the prepared protein PDBQT when prompted."
        )
    if not ligands:
        raise ValueError(
            "No ligand file found for Docking.\n"
            "Please provide the ligand file when prompted."
        )
    if not docking_box:
        raise ValueError(
            "No pocket/docking box found for Docking.\n"
            "Please provide a pocket JSON or enter box coordinates when prompted."
        )

    # Display selected pocket summary before docking starts
    selected_pocket = ctx.get("_selected_pocket")
    selection_method = ctx.get("_selection_method", "best_score")
    if selected_pocket and ctx.get("interactive"):
        display_selected_pocket_summary(selected_pocket, selection_method)

    # Prompt exhaustiveness if interactive
    if ctx.get("interactive"):
        config.docking.exhaustiveness = prompt_exhaustiveness_selection()

    admet_csv = ctx.get("admet_csv")
    if not admet_csv:
        admet_result = ctx.get("_result_admet", {})
        admet_csv = admet_result.get("admet_csv")

    return run_docking(
        protein_pdbqt=protein_pdbqt,
        ligands_path=ligands,
        docking_box=docking_box,
        output_dir=ctx["output_dir"],
        exhaustiveness=config.docking.exhaustiveness,
        n_poses=config.docking.n_poses,
        energy_range=config.docking.energy_range,
        seed=config.docking.seed,
        n_cpus=config.docking.n_cpus,
        score_only=config.docking.score_only,
        admet_csv=admet_csv,
    )



def _exec_lead_selection(ctx: dict, config: DrugLabConfig) -> dict:
    """Execute lead selection module."""
    from drug_selection.lead_selection import select_leads

    # Unified resolution for docking results CSV
    dock_csv = resolve_module_input(ctx, ["results_csv", "docking_results"], [
        f"{ctx.get('output_dir', 'results')}/docking/docking_results.csv"
    ])

    # ADMET results (optional)
    admet_csv = resolve_module_input(ctx, ["admet_csv"], [
        f"{ctx.get('output_dir', 'results')}/admet/admet_results.csv"
    ])

    # Druglikeness results (optional)
    dl_csv = resolve_module_input(ctx, ["descriptors_csv"], [
        f"{ctx.get('output_dir', 'results')}/druglikeness/descriptors.csv"
    ])

    # Ligands SDF for diversity filtering
    ligands_sdf = resolve_module_input(ctx, ["passed_sdf", "combined_sdf"], [
        f"{ctx.get('output_dir', 'results')}/admet/passed_ligands.sdf",
        f"{ctx.get('output_dir', 'results')}/druglikeness/passed_ligands.sdf",
        f"{ctx.get('output_dir', 'results')}/ligand_preparation/prepared_ligands.sdf"
    ])

    if not dock_csv:
        raise ValueError(
            "No docking results CSV found for Lead Selection.\n"
            "Please provide the docking results CSV when prompted."
        )

    res = select_leads(
        docking_results=dock_csv,
        output_dir=ctx["output_dir"],
        admet_results=admet_csv,
        druglikeness_results=dl_csv,
        ligands_sdf=ligands_sdf,
        top_n=config.lead_selection.top_n,
        scoring_method=config.lead_selection.scoring_method,
        weights=config.lead_selection.weights,
        affinity_cutoff=config.lead_selection.affinity_cutoff,
        on_no_hits=config.lead_selection.on_no_hits,
        diversity_filter=config.lead_selection.diversity_filter,
        similarity_cutoff=config.lead_selection.similarity_cutoff,
    )
    
    ctx["selected_lead"] = res.get("best_lead_sdf")
    return res


def _exec_lead_optimization(ctx: dict, config: DrugLabConfig) -> dict:
    """Execute full lead optimization workflow."""
    from lead_optimization.workflow import run_lead_optimization

    print("\nStarting Lead Optimization...")
    print("==================================\n")

    lead_molecule_file = resolve_module_input(ctx, ["lead_molecule_file", "selected_lead", "best_lead_sdf"], [
        f"{ctx.get('output_dir', 'results')}/lead_selection/best_lead.sdf"
    ])

    if not lead_molecule_file:
        raise ValueError("Lead molecule not provided in context.")

    protein_file = resolve_module_input(ctx, ["protein_file", "pdbqt", "clean_pdb", "protein_pdb"], [
        f"{ctx.get('output_dir', 'results')}/protein_preparation/protein.pdbqt"
    ])
    
    if not protein_file:
        raise ValueError("Protein file not provided.")

    # We also attempt to grab docking box if it exists, otherwise workflow orchestrates it
    docking_box = resolve_module_input(ctx, ["docking_box", "docking_box_file"], [
        f"{ctx.get('output_dir', 'results')}/pocket_detection/docking_box.json"
    ])

    return run_lead_optimization(
        lead_file=lead_molecule_file,
        protein_file=protein_file,
        docking_box=docking_box,
        output_dir=ctx["output_dir"],
        config=config,
    )


def create_default_engine() -> WorkflowEngine:
    """Create a WorkflowEngine pre-loaded with all default DrugLab modules."""
    engine = WorkflowEngine()

    engine.register(ModuleNode(
        name="protein_preparation",
        display_name="Protein Preparation",
        menu_id=1,
        dependencies=[],
        inputs=["protein_pdb"],
        outputs=["clean_pdb", "pdbqt"],
        execute_fn=_exec_protein_preparation,
        input_prompts={"protein_pdb": "Protein PDB file: "},
    ))

    engine.register(ModuleNode(
        name="ligand_preparation",
        display_name="Ligand Preparation",
        menu_id=2,
        dependencies=[],
        inputs=["ligand_library"],
        outputs=["combined_sdf"],
        execute_fn=_exec_ligand_preparation,
        input_prompts={"ligand_library": "Ligand library (SDF file or directory): "},
    ))

    engine.register(ModuleNode(
        name="pocket_detection",
        display_name="Pocket Detection",
        menu_id=3,
        dependencies=["protein_preparation"],
        inputs=["protein_pdb"],
        outputs=["docking_box", "selected_pocket"],
        execute_fn=_exec_pocket_detection,
        input_prompts={"protein_pdb": "Protein PDB file: "},
    ))

    engine.register(ModuleNode(
        name="druglikeness",
        display_name="Drug-Likeness Screening",
        menu_id=4,
        dependencies=["ligand_preparation"],
        inputs=["combined_sdf"],
        outputs=["passed_sdf", "descriptors_csv"],
        execute_fn=_exec_druglikeness,
        input_prompts={},
    ))

    engine.register(ModuleNode(
        name="admet",
        display_name="ADMET Screening",
        menu_id=5,
        dependencies=["druglikeness"],
        inputs=["passed_sdf"],
        outputs=["passed_sdf", "admet_csv"],
        execute_fn=_exec_admet,
        input_prompts={"passed_sdf": "Input SDF file for ADMET screening: "},
    ))

    engine.register(ModuleNode(
        name="docking",
        display_name="Docking",
        menu_id=6,
        dependencies=["protein_preparation", "pocket_detection", "admet"],
        inputs=["pdbqt", "passed_sdf", "docking_box"],
        outputs=["results_csv", "best_poses_sdf"],
        execute_fn=_exec_docking,
        input_prompts={},
    ))

    engine.register(ModuleNode(
        name="lead_selection",
        display_name="Lead Selection",
        menu_id=7,
        dependencies=["docking"],
        inputs=["results_csv"],
        outputs=["top_hits_csv", "leads_sdf", "best_lead_sdf"],
        execute_fn=_exec_lead_selection,
        input_prompts={},
    ))

    engine.register(ModuleNode(
        name="lead_optimization",
        display_name="Lead Optimization",
        menu_id=8,
        dependencies=[],
        inputs=["lead_molecule", "protein_pdbqt", "docking_box_file"],
        outputs=[],
        execute_fn=_exec_lead_optimization,
        input_prompts={
            "lead_molecule": "Lead molecule SDF file: ",
            "protein_pdbqt": "Protein PDBQT file: ",
            "docking_box_file": "Docking box JSON file: ",
        },
    ))

    return engine
