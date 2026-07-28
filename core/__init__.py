"""
Core package for DrugLab. Provides submodules for protein preparation, ligand preparation, pocket detection, drug-likeness, ADMET, and docking.
"""


# Expose submodules
from importlib import import_module

def __getattr__(name):
    try:
        return import_module(f'druglab.core.{name}')
    except ModuleNotFoundError as e:
        raise AttributeError(f"module 'core' has no attribute '{name}'") from e
