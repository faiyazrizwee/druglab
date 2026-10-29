"""
DrugLab Fetch Utility.

Provides functions to retrieve protein structures from RCSB PDB and
chemical structures (SDF) from the PubChem REST API.
"""

import os
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
from utils.logger import setup_logger, log_step
from utils.io import ensure_dir

logger = setup_logger("fetch", level="INFO")

RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
PUBCHEM_NAME_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/SDF"
PUBCHEM_CID_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF"


def fetch_pdb(pdb_id: str, output_path: str | None = None) -> str:
    """
    Download a PDB structure by its 4-character RCSB PDB identifier.

    Args:
        pdb_id: RCSB PDB code (e.g., '7DTD', '4MZI', '6LU7').
        output_path: Optional destination file path. If None, saves to
                     protein/{pdb_id}.pdb.

    Returns:
        The path of the downloaded PDB file.
    """
    clean_id = pdb_id.strip().upper()
    if len(clean_id) != 4:
        raise ValueError(f"Invalid PDB ID '{pdb_id}'. PDB IDs must be 4 characters.")

    if output_path is None:
        output_path = os.path.join("protein", f"{clean_id}.pdb")

    ensure_dir(os.path.dirname(os.path.abspath(output_path)))
    url = RCSB_PDB_URL.format(pdb_id=clean_id)

    log_step(logger, f"Fetching PDB {clean_id}", "started", {"url": url})
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "DrugLab/0.2.0 (Computational Drug Discovery)"}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read().decode("utf-8", errors="replace")

        if "HEADER" not in content and "ATOM" not in content:
            raise ValueError(f"Downloaded content from {url} is not a valid PDB file.")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        log_step(logger, f"Fetching PDB {clean_id}", "completed", {
            "path": output_path,
            "bytes": len(content),
        })
        return output_path

    except urllib.error.HTTPError as e:
        logger.error(f"HTTP Error {e.code} fetching PDB {clean_id} from {url}: {e.reason}")
        raise
    except Exception as e:
        logger.error(f"Failed to fetch PDB {clean_id}: {e}")
        raise


def fetch_ligand_by_name(name: str, output_path: str | None = None) -> str:
    """
    Download a chemical structure in SDF format from PubChem by common or chemical name.

    Args:
        name: Compound name (e.g., 'apigenin', 'quercetin', 'aspirin').
        output_path: Optional destination file path. If None, saves to
                     ligands/{safe_name}.sdf.

    Returns:
        The path of the downloaded SDF file.
    """
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Ligand name cannot be empty.")

    safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in clean_name.lower())

    if output_path is None:
        output_path = os.path.join("ligands", f"{safe_name}.sdf")

    ensure_dir(os.path.dirname(os.path.abspath(output_path)))
    url = PUBCHEM_NAME_URL.format(name=urllib.parse.quote(clean_name))

    log_step(logger, f"Fetching ligand '{clean_name}'", "started", {"url": url})
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "DrugLab/0.2.0 (Computational Drug Discovery)"}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read().decode("utf-8", errors="replace")

        if "$$$$" not in content and "M  END" not in content:
            raise ValueError(f"PubChem response for '{clean_name}' is not a valid SDF.")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        log_step(logger, f"Fetching ligand '{clean_name}'", "completed", {
            "path": output_path,
            "bytes": len(content),
        })
        return output_path

    except urllib.error.HTTPError as e:
        logger.error(f"HTTP Error {e.code} fetching ligand '{clean_name}': {e.reason}")
        raise
    except Exception as e:
        logger.error(f"Failed to fetch ligand '{clean_name}': {e}")
        raise


def fetch_ligand_by_cid(cid: int | str, output_path: str | None = None) -> str:
    """
    Download a chemical structure in SDF format from PubChem by PubChem Compound ID (CID).

    Args:
        cid: PubChem CID integer or string (e.g., 5280443 for apigenin).
        output_path: Optional destination file path. If None, saves to
                     ligands/CID_{cid}.sdf.

    Returns:
        The path of the downloaded SDF file.
    """
    clean_cid = str(cid).strip()
    if not clean_cid.isdigit():
        raise ValueError(f"Invalid PubChem CID '{cid}'. CID must be a positive integer.")

    if output_path is None:
        output_path = os.path.join("ligands", f"CID_{clean_cid}.sdf")

    ensure_dir(os.path.dirname(os.path.abspath(output_path)))
    url = PUBCHEM_CID_URL.format(cid=clean_cid)

    log_step(logger, f"Fetching PubChem CID {clean_cid}", "started", {"url": url})
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "DrugLab/0.2.0 (Computational Drug Discovery)"}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read().decode("utf-8", errors="replace")

        if "$$$$" not in content and "M  END" not in content:
            raise ValueError(f"PubChem response for CID {clean_cid} is not a valid SDF.")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        log_step(logger, f"Fetching PubChem CID {clean_cid}", "completed", {
            "path": output_path,
            "bytes": len(content),
        })
        return output_path

    except urllib.error.HTTPError as e:
        logger.error(f"HTTP Error {e.code} fetching CID {clean_cid}: {e.reason}")
        raise
    except Exception as e:
        logger.error(f"Failed to fetch CID {clean_cid}: {e}")
        raise
