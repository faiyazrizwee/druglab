"""
DrugLab Fetch Utility.

Provides functions to retrieve protein structures from RCSB PDB and
chemical structures (SDF) from the PubChem REST API.
"""

import os
import re
import json
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
from typing import Any
from utils.logger import setup_logger, log_step
from utils.io import ensure_dir

logger = setup_logger("fetch", level="INFO")

RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
PUBCHEM_NAME_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/SDF"
PUBCHEM_CID_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF"
ALPHAFOLD_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"
ALPHAFOLD_ENTRY_URL_PREFIX = "https://alphafold.ebi.ac.uk/entry/"

# Standard UniProt accession regex patterns
# Matches 6-character (e.g. Q9KPW0, P04637) and 10-character (e.g. A0A024RBG1) UniProt accessions
UNIPROT_REGEX = re.compile(
    r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9][A-Z0-9]{3}[0-9][A-Z0-9]{4}$",
    re.IGNORECASE,
)


def extract_uniprot_accession(query: str) -> str | None:
    """
    Extract a clean UniProt accession from a raw accession string or an AlphaFold / UniProt URL.

    Supported formats:
        - Raw accession: 'Q9KPW0', 'P04637', 'A0A024RBG1'
        - AlphaFold URL: 'https://alphafold.ebi.ac.uk/entry/Q9KPW0'
        - AlphaFold URL with trailing slash: 'https://alphafold.ebi.ac.uk/entry/Q9KPW0/'
        - UniProt URL: 'https://www.uniprot.org/uniprotkb/Q9KPW0/entry'
        - UniProt URL: 'https://www.uniprot.org/uniprotkb/Q9KPW0'

    Args:
        query: Accession string or URL.

    Returns:
        The extracted uppercase UniProt accession, or None if no valid accession is found.
    """
    if not query:
        return None

    cleaned = query.strip()

    # Handle AlphaFold entry URLs: https://alphafold.ebi.ac.uk/entry/{accession}
    af_match = re.search(r"alphafold\.ebi\.ac\.uk/entry/([A-Za-z0-9_-]+)", cleaned, re.IGNORECASE)
    if af_match:
        cand = af_match.group(1).strip().upper()
        # Strip trailing subpaths or parameters if present
        cand = cand.split("/")[0].split("?")[0]
        if UNIPROT_REGEX.match(cand):
            return cand

    # Handle UniProt URLs: https://www.uniprot.org/uniprotkb/{accession}
    uniprot_match = re.search(r"uniprot\.org/uniprotkb/([A-Za-z0-9_-]+)", cleaned, re.IGNORECASE)
    if uniprot_match:
        cand = uniprot_match.group(1).strip().upper()
        cand = cand.split("/")[0].split("?")[0]
        if UNIPROT_REGEX.match(cand):
            return cand

    # Check if the query itself is a valid UniProt accession
    cand = cleaned.upper()
    if UNIPROT_REGEX.match(cand):
        return cand

    # Fallback search for a standalone UniProt accession token inside text
    tokens = re.findall(r"\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z0-9]{3}[0-9][A-Z0-9]{4})\b", cleaned, re.IGNORECASE)
    for tok in tokens:
        if UNIPROT_REGEX.match(tok.upper()):
            return tok.upper()

    return None


def validate_uniprot_accession(accession: str) -> bool:
    """Check if the provided string is a valid UniProt accession format."""
    if not accession:
        return False
    return bool(UNIPROT_REGEX.match(accession.strip().upper()))


def fetch_alphafold(
    accession_or_url: str,
    output_path: str | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """
    Fetch a predicted protein 3D structure from the AlphaFold Database (EMBL-EBI / DeepMind)
    by UniProt accession or AlphaFold entry URL.

    Args:
        accession_or_url: UniProt accession (e.g. 'Q9KPW0') or full AlphaFold URL
                          (e.g. 'https://alphafold.ebi.ac.uk/entry/Q9KPW0').
        output_path: Optional destination file path. If None, saves to
                     protein/AF_{accession}.pdb.
        timeout: HTTP request timeout in seconds.

    Returns:
        Dict containing metadata, pLDDT metrics, and the path to the downloaded PDB structure.

    Raises:
        ValueError: If accession is invalid, missing, or not found in AlphaFold DB.
        RuntimeError: If network error or download fails.
    """
    accession = extract_uniprot_accession(accession_or_url)
    if not accession:
        raise ValueError(
            f"Invalid UniProt accession or AlphaFold URL: '{accession_or_url}'.\n"
            f"Please provide a valid UniProt accession (e.g. 'Q9KPW0', 'P04637') "
            f"or an AlphaFold entry URL (e.g. 'https://alphafold.ebi.ac.uk/entry/Q9KPW0')."
        )

    if output_path is None:
        output_path = os.path.join("protein", f"AF_{accession}.pdb")

    ensure_dir(os.path.dirname(os.path.abspath(output_path)))
    api_url = ALPHAFOLD_API_URL.format(accession=accession)

    log_step(logger, f"Querying AlphaFold DB for '{accession}'", "started", {"api_url": api_url})

    try:
        req = urllib.request.Request(
            api_url,
            headers={
                "User-Agent": "DrugLab/0.2.0 (Computational Drug Discovery; mailto:contact@druglab.local)",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw_data = response.read().decode("utf-8", errors="replace")
            entries = json.loads(raw_data)

    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise ValueError(
                f"Protein accession '{accession}' was not found in the AlphaFold Database (HTTP 404).\n"
                f"Please verify that the UniProt accession is correct and that a predicted structure "
                f"has been deposited in AlphaFold DB."
            ) from e
        elif e.code == 400:
            raise ValueError(
                f"AlphaFold Database rejected the query for '{accession}' (HTTP 400).\n"
                f"Please verify the format of the accession."
            ) from e
        else:
            raise RuntimeError(
                f"AlphaFold API returned HTTP error {e.code} ({e.reason}) for '{accession}'."
            ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Network error connecting to AlphaFold Database at {api_url}: {e.reason}.\n"
            f"Please verify your internet connection or proxy settings."
        ) from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error while querying AlphaFold Database for '{accession}': {e}") from e

    if not entries or not isinstance(entries, list):
        raise ValueError(f"No AlphaFold prediction records returned for accession '{accession}'.")

    entry = entries[0]
    pdb_url = entry.get("pdbUrl")
    if not pdb_url:
        raise ValueError(f"AlphaFold entry for '{accession}' does not contain an available PDB download URL.")

    # Download PDB file
    log_step(logger, f"Downloading AlphaFold PDB for '{accession}'", "started", {"pdb_url": pdb_url})
    try:
        pdb_req = urllib.request.Request(
            pdb_url,
            headers={"User-Agent": "DrugLab/0.2.0 (Computational Drug Discovery)"},
        )
        with urllib.request.urlopen(pdb_req, timeout=timeout) as response:
            pdb_content = response.read().decode("utf-8", errors="replace")

        if "ATOM" not in pdb_content:
            raise ValueError(f"Downloaded AlphaFold content from {pdb_url} does not contain valid PDB coordinates.")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(pdb_content)

    except Exception as e:
        raise RuntimeError(f"Failed to download AlphaFold structure from {pdb_url}: {e}") from e

    # Extract confidence metrics from API response
    mean_plddt = float(entry.get("globalMetricValue") or 0.0)
    frac_vh = float(entry.get("fractionPlddtVeryHigh") or 0.0)
    frac_c = float(entry.get("fractionPlddtConfident") or 0.0)
    frac_l = float(entry.get("fractionPlddtLow") or 0.0)
    frac_vl = float(entry.get("fractionPlddtVeryLow") or 0.0)

    # If globalMetricValue was not provided by API, calculate mean pLDDT directly from B-factors in PDB
    if mean_plddt <= 0.0:
        plddt_vals = []
        for line in pdb_content.splitlines():
            if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 66:
                try:
                    plddt_vals.append(float(line[60:66].strip()))
                except ValueError:
                    pass
        if plddt_vals:
            mean_plddt = sum(plddt_vals) / len(plddt_vals)
            frac_vh = sum(1 for v in plddt_vals if v >= 90) / len(plddt_vals)
            frac_c = sum(1 for v in plddt_vals if 70 <= v < 90) / len(plddt_vals)
            frac_l = sum(1 for v in plddt_vals if 50 <= v < 70) / len(plddt_vals)
            frac_vl = sum(1 for v in plddt_vals if v < 50) / len(plddt_vals)

    low_conf_pct = round((frac_l + frac_vl) * 100, 1)

    result = {
        "accession": accession,
        "entry_id": entry.get("entryId", f"AF-{accession}-F1"),
        "gene": entry.get("gene") or "N/A",
        "uniprot_id": entry.get("uniprotId") or f"{accession}_UNKNOWN",
        "description": entry.get("uniprotDescription") or "Predicted protein structure",
        "organism": entry.get("organismScientificName") or "Unknown organism",
        "latest_version": entry.get("latestVersion", 4),
        "sequence_length": len(entry.get("sequence", "")),
        "pdb_url": pdb_url,
        "cif_url": entry.get("cifUrl", ""),
        "output_path": output_path,
        "is_predicted": True,
        "is_alphafold": True,
        "source": f"AlphaFold DB v{entry.get('latestVersion', 4)} (EMBL-EBI / DeepMind)",
        "mean_plddt": round(mean_plddt, 1),
        "fraction_very_high": round(frac_vh, 3),
        "fraction_confident": round(frac_c, 3),
        "fraction_low": round(frac_l, 3),
        "fraction_very_low": round(frac_vl, 3),
        "plddt_very_high_pct": round(frac_vh * 100, 1),
        "plddt_confident_pct": round(frac_c * 100, 1),
        "plddt_low_pct": round(frac_l * 100, 1),
        "plddt_very_low_pct": round(frac_vl * 100, 1),
        "low_confidence_warning": low_conf_pct > 20.0,
        "status": "completed",
    }

    log_step(logger, f"AlphaFold structure '{accession}' fetched", "completed", {
        "path": output_path,
        "mean_plddt": result["mean_plddt"],
        "gene": result["gene"],
        "organism": result["organism"],
    })
    return result


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
