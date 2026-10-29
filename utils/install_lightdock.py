#!/usr/bin/env python3
"""
LightDock 0.9.4 NumPy 2.x & GCC 14+ Installer & Patcher for DrugLab.

This script resolves upstream LightDock 0.9.4 C-extension compatibility issues:
1. In cdfire2.c, PyArray_DATA(PyObject*) calls are patched with explicit
   (PyArrayObject*) casts required by NumPy 2.0+ and strict GCC 14+ typing.
2. In setup.py, deprecated 'nose' is removed from setup_requires.
3. Builds and installs LightDock cleanly into the active Python environment.
"""

import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request


LIGHTDOCK_URL = "https://files.pythonhosted.org/packages/source/l/lightdock/lightdock-0.9.4.tar.gz"


def patch_lightdock_source(source_dir: str) -> None:
    """Apply NumPy 2.x and GCC 14+ patches to LightDock source."""
    cdfire2_path = os.path.join(source_dir, "lightdock", "scoring", "dfire2", "c", "cdfire2.c")
    if os.path.exists(cdfire2_path):
        with open(cdfire2_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Add explicit PyArrayObject* casts to PyArray_DATA calls
        patched = content.replace(
            "res_indexes = (int*)PyArray_DATA(res_array);",
            "res_indexes = (int*)PyArray_DATA((PyArrayObject*)res_array);",
        )
        patched = patched.replace(
            "atom_indexes = (int*)PyArray_DATA(atom_array);",
            "atom_indexes = (int*)PyArray_DATA((PyArrayObject*)atom_array);",
        )
        patched = patched.replace(
            "energies = (double*)PyArray_DATA(energy_array);",
            "energies = (double*)PyArray_DATA((PyArrayObject*)energy_array);",
        )

        with open(cdfire2_path, "w", encoding="utf-8") as f:
            f.write(patched)
        print("  ✓ Patched cdfire2.c (NumPy 2.x PyArrayObject* casts)")

    setup_path = os.path.join(source_dir, "setup.py")
    if os.path.exists(setup_path):
        with open(setup_path, "r", encoding="utf-8") as f:
            setup_content = f.read()

        # Remove deprecated nose from setup_requires
        setup_patched = setup_content.replace(
            'setup_requires=["numpy>=1.17.1", "nose"]',
            'setup_requires=["numpy>=1.17.1"]',
        )
        with open(setup_path, "w", encoding="utf-8") as f:
            f.write(setup_patched)
        print("  ✓ Patched setup.py (removed deprecated nose requirement)")


def install_lightdock() -> int:
    """Download, patch, and install LightDock 0.9.4."""
    print("=" * 60)
    print("DrugLab: Installing LightDock 0.9.4 with NumPy 2.x / GCC 14+ Fix")
    print("=" * 60)

    # Ensure build tools are present
    build_deps = ["wheel", "setuptools", "cython", "numpy"]
    for dep in build_deps:
        try:
            __import__(dep)
        except ImportError:
            print(f"Installing missing build dependency: {dep}...")
            subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)

    with tempfile.TemporaryDirectory(prefix="druglab_lightdock_") as tmpdir:
        tarball_path = os.path.join(tmpdir, "lightdock-0.9.4.tar.gz")
        print(f"Downloading LightDock 0.9.4 from PyPI...")
        urllib.request.urlretrieve(LIGHTDOCK_URL, tarball_path)

        print("Extracting source archive...")
        with tarfile.open(tarball_path, "r:gz") as tar:
            tar.extractall(tmpdir)

        source_dir = os.path.join(tmpdir, "lightdock-0.9.4")
        if not os.path.isdir(source_dir):
            # Check for alternative directory structure
            subdirs = [d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))]
            source_dir = os.path.join(tmpdir, subdirs[0])

        print("Applying compatibility patches...")
        patch_lightdock_source(source_dir)

        print("Compiling and installing LightDock...")
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "--no-deps",
            source_dir,
        ]
        result = subprocess.run(cmd)

        if result.returncode == 0:
            print("\n✓ Successfully installed LightDock 0.9.4 for DrugLab!")
            # Quick verification
            try:
                import lightdock
                from lightdock.version import CURRENT_VERSION
                print(f"  Verified import: LightDock {CURRENT_VERSION} ({lightdock.__file__})")
            except Exception as e:
                print(f"  Warning during import check: {e}")
            return 0
        else:
            print(f"\n✗ LightDock installation failed with exit code {result.returncode}")
            return result.returncode


if __name__ == "__main__":
    sys.exit(install_lightdock())
