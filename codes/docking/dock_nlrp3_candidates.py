#!/usr/bin/env python3
"""Dock three selected repurposing candidates against NLRP3 with AutoDock Vina."""
from __future__ import annotations
import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_NAME = "NLRP3"
PDB_ID = "9GU4"
RECEPTOR_CHAINS: Optional[List[str]] = ["A"]
NATIVE_LIGAND_RESNAME = "A1IPJ"
NATIVE_LIGAND_CHAIN: Optional[str] = "A"
NATIVE_LIGAND_DISPLAY_NAME = "NP3-253"
NATIVE_LIGAND_PDB_ALIAS = "NP3"
STRUCTURE_FILE_FORMAT = "mmcif"
BOX_PADDING = 8.0
MIN_BOX_SIZE = (24.0, 24.0, 24.0)
MAX_BOX_SIZE = (32.0, 32.0, 32.0)
BOX_SIZE_OVERRIDE: Optional[Tuple[float, float, float]] = None
BOX_CENTER_OVERRIDE: Optional[Tuple[float, float, float]] = None
EXHAUSTIVENESS = 32
NUM_MODES = 10
ENERGY_RANGE = 4
CPU = 8
SEED = 2026
VINA_TIMEOUT_SECONDS = 0
# Compounds reported in the manuscript's final NLRP3 docking analysis.
HARDCODED_COMPOUNDS: Tuple[Dict[str, object], ...] = (
    {
        "rank": 4,
        "chembl_id": "CHEMBL1372950",
        "compound_name": "NICERGOLINE",
        "smiles": "CO[C@]12C[C@@H](COC(=O)c3cncc(Br)c3)CN(C)[C@@H]1Cc1cn(C)c3cccc2c13",
    },
    {
        "rank": 5,
        "chembl_id": "CHEMBL267495",
        "compound_name": "NALFURAFINE",
        "smiles": "CN(C(=O)/C=C/c1ccoc1)[C@@H]1CC[C@@]2(O)[C@H]3Cc4ccc(O)c5c4[C@@]2(CCN3CC2CC2)[C@H]1O5",
    },
    {
        "rank": 7,
        "chembl_id": "CHEMBL4650827",
        "compound_name": "REVUMENIB",
        "smiles": "CCN(C(=O)c1cc(F)ccc1Oc1cncnc1N1CC2(CCN(C[C@H]3CC[C@H](NS(=O)(=O)CC)CC3)CC2)C1)C(C)C",
    },
)
LIGAND_PH = 7.4
RECEPTOR_PREP_BACKEND = "autodocktools"
AUTODOCKTOOLS_CONDA_ENV = "mgltools"
AUTODOCKTOOLS_EXPLICIT_PREPARE_RECEPTOR4: Optional[str] = None
AUTODOCKTOOLS_REPAIRS = "hydrogens"
AUTODOCKTOOLS_CLEANUP = "nphs_lps_waters_nonstdres"
AUDIT_RECEPTOR_MISSING_ATOMS = True
POCKET_INCOMPLETE_RESIDUE_WARNING_DISTANCE = 6.0
FAIL_IF_INCOMPLETE_RESIDUE_NEAR_POCKET = False
REPAIR_RECEPTOR_WITH_PDBFIXER = False
RUN_RESTRAINED_MINIMIZATION = False
SPLIT_RECEPTOR_AT_COORDINATE_GAPS = True
REPAIR_RECEPTOR_WITH_PDBFIXER = False
REPAIR_PH = 7.4
REPAIR_ADD_HYDROGENS = False
REPAIR_REPLACE_NONSTANDARD_RESIDUES = True
REPAIR_FAIL_IF_HEAVY_ATOMS_REMAIN_MISSING = True
REPAIR_PIPELINE_VERSION = "gap-aware-minimized-v3"
SPLIT_RECEPTOR_AT_COORDINATE_GAPS = True
PEPTIDE_BOND_MAX_DISTANCE = 1.9
CHAIN_BREAK_POCKET_WARNING_DISTANCE = 8.0
FAIL_IF_CHAIN_BREAK_NEAR_POCKET = False
RUN_RESTRAINED_MINIMIZATION = False
MINIMIZATION_FORCEFIELD_CANDIDATES = [
    ("amber14-all.xml", "implicit/obc2.xml"),
    ("amber99sb.xml", "amber99_obc.xml"),
]
MINIMIZATION_MAX_ITERATIONS = 2500
MINIMIZATION_RETRY_MAX_ITERATIONS = 6000
MINIMIZATION_TOLERANCE_KJ_PER_MOL_NM = 10.0
MINIMIZATION_BACKBONE_RESTRAINT_K = 1500.0
MINIMIZATION_SIDECHAIN_RESTRAINT_K = 300.0
MINIMIZATION_REPAIRED_SIDECHAIN_RESTRAINT_K = 50.0
MINIMIZATION_RETRY_RESTRAINT_SCALE = 0.2
MINIMIZATION_PLATFORM_PREFERENCE = ("CPU", "Reference")
FAIL_IF_RDKIT_VALENCE_AUDIT_FAILS = True
STRIP_RECEPTOR_HYDROGENS_BEFORE_MEEKO = True
MEEKO_RECEPTOR_PARSER = "auto"
ALLOW_BAD_RESIDUES = False
OVERWRITE_STRUCTURE_DOWNLOAD = False
RESUME = True
CREATE_VISUALIZATION_FILES = True
RUN_NATIVE_LIGAND_REDOCK = False


@dataclass
class LigandRecord:
    input_row: int
    rank: Optional[int]
    chembl_id: str
    compound_name: str
    smiles: str
    safe_name: str = ""


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sanitize_name(name: str, max_length: int = 100) -> str:
    text = str(name or "").strip()
    text = re.sub("\\s+", "_", text)
    text = re.sub("[^A-Za-z0-9_.-]", "_", text)
    text = re.sub("_+", "_", text).strip("_.")
    if not text:
        text = "compound"
    return text[:max_length]


def build_hardcoded_ligands() -> List[LigandRecord]:
    records: List[LigandRecord] = []
    used_safe_names: Set[str] = set()
    for input_row, item in enumerate(HARDCODED_COMPOUNDS, start=1):
        rank = int(item["rank"])
        chembl_id = str(item["chembl_id"]).strip()
        compound_name = str(item["compound_name"]).strip()
        smiles = str(item["smiles"]).strip()
        if not chembl_id or not compound_name or (not smiles):
            raise RuntimeError(f"Incomplete hardcoded compound record: {item}")
        base = sanitize_name(f"{rank:04d}_{chembl_id}_{compound_name}")
        safe_name = base
        serial = 2
        while safe_name in used_safe_names:
            safe_name = f"{base}_{serial}"
            serial += 1
        used_safe_names.add(safe_name)
        records.append(
            LigandRecord(
                input_row=input_row,
                rank=rank,
                chembl_id=chembl_id,
                compound_name=compound_name,
                smiles=smiles,
                safe_name=safe_name,
            )
        )
    print(
        f"[OK] Loaded {len(records)} compounds from the built-in list; no spreadsheet is required."
    )
    return records


def read_log_tail(log_file: Optional[Path], n_lines: int = 50) -> str:
    if log_file is None or not log_file.exists():
        return ""
    try:
        lines = log_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(lines[-n_lines:])
    except Exception:
        return ""


def run_cmd(
    cmd: Sequence[str],
    cwd: Optional[Path] = None,
    log_file: Optional[Path] = None,
    timeout_seconds: int = 0,
) -> subprocess.CompletedProcess:
    cmd = [str(x) for x in cmd]
    print("\n[CMD]", " ".join(cmd), flush=True)
    timeout = timeout_seconds if timeout_seconds and timeout_seconds > 0 else None
    try:
        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "w", encoding="utf-8", errors="ignore") as f:
                process = subprocess.run(
                    cmd,
                    cwd=str(cwd) if cwd else None,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
        else:
            process = subprocess.run(
                cmd,
                cwd=str(cwd) if cwd else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
                check=False,
            )
            if process.stdout:
                print(process.stdout, flush=True)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Command timed out after {timeout_seconds} seconds: {' '.join(cmd)}"
        ) from exc
    if process.returncode != 0:
        tail = read_log_tail(log_file)
        extra = f"\nLog tail:\n{tail}" if tail else ""
        raise RuntimeError(
            f"Command failed with return code {process.returncode}：{' '.join(cmd)}{extra}"
        )
    return process


def check_command_exists(command: str) -> Path:
    path = shutil.which(command)
    if path is None:
        raise RuntimeError(
            f"Required external command was not found: {command}\nActivate the autodock_vina environment and install Vina, Meeko, and Open Babel."
        )
    p = Path(path)
    print(f"[OK] {command}: {p}")
    return p


def check_python_package(package: str, install_hint: str = "") -> None:
    try:
        __import__(package)
    except ImportError as exc:
        hint = f"\nInstallation example: {install_hint}" if install_hint else ""
        raise RuntimeError(f"Missing Python package: {package}{hint}") from exc


def download_file(url: str, out_file: Path, overwrite: bool = False) -> None:
    if out_file.exists() and out_file.stat().st_size > 0 and (not overwrite):
        print(f"[SKIP] Existing file: {out_file}")
        return
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = out_file.with_suffix(out_file.suffix + ".part")
    if tmp_file.exists():
        tmp_file.unlink()
    print(f"[DOWNLOAD] {url}\n           -> {out_file}")
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 NLRP3-Vina-Pipeline/1.0"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            with open(tmp_file, "wb") as f:
                shutil.copyfileobj(response, f)
        tmp_file.replace(out_file)
    except Exception as exc:
        if tmp_file.exists():
            tmp_file.unlink()
        raise RuntimeError(f"Download failed: {url}\nError: {exc}") from exc
    if not out_file.exists() or out_file.stat().st_size == 0:
        raise RuntimeError(f"Downloaded file is empty: {out_file}")


def atomic_write_json(data: Dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(output)


def parse_pdb_atom_coord(line: str) -> Tuple[float, float, float]:
    return (float(line[30:38]), float(line[38:46]), float(line[46:54]))


def _gemmi_altloc_text(atom) -> str:
    value = str(getattr(atom, "altloc", "") or "")
    return value.replace("\\x00", "").replace("\x00", "").strip()


def _choose_residue_atoms(residue) -> List[object]:
    by_name: Dict[str, List[object]] = {}
    for atom in residue:
        by_name.setdefault(str(atom.name).strip(), []).append(atom)
    selected: List[object] = []
    for atom_name, atoms in by_name.items():

        def priority(atom) -> Tuple[int, float]:
            altloc = _gemmi_altloc_text(atom).upper()
            if altloc == "":
                alt_rank = 0
            elif altloc == "A":
                alt_rank = 1
            else:
                alt_rank = 2
            return (alt_rank, -float(getattr(atom, "occ", 0.0) or 0.0))

        chosen = sorted(atoms, key=priority)[0].clone()
        try:
            chosen.altloc = "\x00"
        except Exception:
            pass
        selected.append(chosen)
    return selected


def _build_single_residue_pdb(
    source_structure, source_residue, output_pdb: Path, pdb_resname: str
) -> None:
    import gemmi

    out = gemmi.Structure()
    out.name = f"{PDB_ID}_{pdb_resname}"
    out.cell = source_structure.cell
    out.spacegroup_hm = source_structure.spacegroup_hm
    model = gemmi.Model("1")
    chain = gemmi.Chain("Z")
    residue = gemmi.Residue()
    residue.name = pdb_resname[:3].upper()
    residue.seqid = gemmi.SeqId(1, " ")
    residue.het_flag = "H"
    for atom in _choose_residue_atoms(source_residue):
        chain_atom = atom.clone()
        try:
            chain_atom.altloc = "\x00"
        except Exception:
            pass
        residue.add_atom(chain_atom)
    chain.add_residue(residue)
    model.add_chain(chain)
    out.add_model(model)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    out.write_pdb(str(output_pdb))


def extract_receptor_and_native_ligand_from_mmcif(
    structure_file: Path,
    receptor_pdb: Path,
    native_ligand_pdb: Path,
    receptor_chains: Optional[List[str]],
    ligand_ccd_id: str,
    ligand_chain: Optional[str],
    ligand_pdb_alias: str,
) -> Dict[str, object]:
    try:
        import gemmi
    except ImportError as exc:
        raise RuntimeError(
            "Gemmi is required. Install it with: conda install -c conda-forge gemmi"
        ) from exc
    if not structure_file.exists():
        raise FileNotFoundError(
            f"mmCIF structure file does not exist: {structure_file}"
        )
    structure = gemmi.read_structure(str(structure_file))
    if len(structure) == 0:
        raise RuntimeError(f"No model was found in the mmCIF file: {structure_file}")
    try:
        structure.setup_entities()
        structure.assign_het_flags()
    except Exception:
        pass
    model_in = structure[0]
    chain_filter = (
        None if receptor_chains is None else {str(x).strip() for x in receptor_chains}
    )
    ligand_ccd_id = ligand_ccd_id.upper().strip()
    visible_nonpolymer: List[Dict[str, object]] = []
    ligand_candidates: List[Tuple[object, object]] = []
    for chain in model_in:
        for residue in chain:
            is_polymer_atom = str(getattr(residue, "het_flag", "")).upper() == "A"
            if not is_polymer_atom:
                visible_nonpolymer.append(
                    {
                        "chain": str(chain.name),
                        "subchain": str(getattr(residue, "subchain", "")),
                        "resname": str(residue.name),
                        "seqid": str(residue.seqid),
                        "atom_count": len(list(residue)),
                    }
                )
            if str(residue.name).upper() == ligand_ccd_id:
                if ligand_chain is None or str(chain.name) == str(ligand_chain):
                    ligand_candidates.append((chain, residue))
    if not ligand_candidates:
        preview = visible_nonpolymer[:50]
        raise RuntimeError(
            f"Could not extract {structure_file}find the co-crystal ligand with CCD ID {ligand_ccd_id}，ligand_chain={ligand_chain}.\nFirst 50 non-polymer residues in the mmCIF file:\n{json.dumps(preview, ensure_ascii=False, indent=2)}"
        )
    ligand_candidates.sort(key=lambda item: len(list(item[1])), reverse=True)
    ligand_chain_obj, ligand_residue = ligand_candidates[0]
    out = gemmi.Structure()
    out.name = f"{TARGET_NAME}_{PDB_ID}_protein"
    out.cell = structure.cell
    out.spacegroup_hm = structure.spacegroup_hm
    out_model = gemmi.Model("1")
    protein_residue_count = 0
    protein_atom_count = 0
    chains_found: List[str] = []
    for chain in model_in:
        chain_name = str(chain.name)
        if chain_filter is not None and chain_name not in chain_filter:
            continue
        out_chain = gemmi.Chain(chain_name or "A")
        chain_residue_count = 0
        for residue in chain:
            if str(getattr(residue, "het_flag", "")).upper() != "A":
                continue
            new_residue = gemmi.Residue()
            new_residue.name = str(residue.name)
            new_residue.seqid = residue.seqid
            new_residue.het_flag = "A"
            try:
                new_residue.subchain = str(getattr(residue, "subchain", ""))
                new_residue.label_seq = int(getattr(residue, "label_seq", 0) or 0)
            except Exception:
                pass
            selected_atoms = _choose_residue_atoms(residue)
            if not selected_atoms:
                continue
            for atom in selected_atoms:
                new_residue.add_atom(atom)
                protein_atom_count += 1
            out_chain.add_residue(new_residue)
            protein_residue_count += 1
            chain_residue_count += 1
        if chain_residue_count > 0:
            out_model.add_chain(out_chain)
            chains_found.append(chain_name)
    if protein_atom_count == 0:
        raise RuntimeError(
            f"Could not extract {structure_file}protein polymer atoms. Check RECEPTOR_CHAINS={receptor_chains}。"
        )
    out.add_model(out_model)
    receptor_pdb.parent.mkdir(parents=True, exist_ok=True)
    out.write_pdb(str(receptor_pdb))
    _build_single_residue_pdb(
        source_structure=structure,
        source_residue=ligand_residue,
        output_pdb=native_ligand_pdb,
        pdb_resname=ligand_pdb_alias,
    )
    ligand_instances = [
        {
            "auth_chain": str(chain.name),
            "subchain": str(getattr(residue, "subchain", "")),
            "seqid": str(residue.seqid),
            "ccd_id": str(residue.name),
            "atom_count": len(_choose_residue_atoms(residue)),
        }
        for chain, residue in ligand_candidates
    ]
    chosen_atom_count = len(_choose_residue_atoms(ligand_residue))
    if chosen_atom_count < 10:
        raise RuntimeError(
            f"Extracted {ligand_ccd_id} has an unexpectedly low atom count: {chosen_atom_count}. Check the mmCIF content and ligand CCD identifier."
        )
    print(f"[OK] receptor PDB：{receptor_pdb}")
    print(f"[OK] native ligand PDB：{native_ligand_pdb}")
    print(
        f"[INFO] mmCIF={structure_file.name}, receptor chains={chains_found}, protein residues={protein_residue_count}, protein atoms={protein_atom_count}"
    )
    print(
        f"[INFO] native ligand={NATIVE_LIGAND_DISPLAY_NAME}, CCD={ligand_ccd_id}, chosen auth_chain={ligand_chain_obj.name}, atoms={chosen_atom_count}"
    )
    return {
        "structure_format": "mmCIF",
        "structure_file": str(structure_file),
        "receptor_chains_requested": receptor_chains,
        "receptor_chains_found": chains_found,
        "protein_residue_count": protein_residue_count,
        "protein_atom_count": protein_atom_count,
        "native_ligand_display_name": NATIVE_LIGAND_DISPLAY_NAME,
        "native_ligand_ccd_id": ligand_ccd_id,
        "native_ligand_pdb_alias": ligand_pdb_alias,
        "native_ligand_instances": ligand_instances,
        "selected_native_ligand_atom_count": chosen_atom_count,
        "visible_nonpolymer_residues": visible_nonpolymer,
    }


def _pdb_element_from_line(line: str) -> str:
    element = line[76:78].strip().upper() if len(line) >= 78 else ""
    if element:
        return element
    atom_name = line[12:16].strip().upper() if len(line) >= 16 else ""
    atom_name = re.sub("^[0-9]+", "", atom_name)
    if atom_name.startswith(("CL", "BR")):
        return atom_name[:2]
    return atom_name[:1]


def _pdb_residue_key(line: str) -> Tuple[str, int, str, str]:
    chain_id = line[21].strip() if len(line) > 21 else ""
    try:
        residue_number = int(line[22:26])
    except Exception:
        residue_number = 0
    insertion_code = line[26].strip() if len(line) > 26 else ""
    residue_name = line[17:20].strip().upper() if len(line) >= 20 else "UNK"
    return (chain_id, residue_number, insertion_code, residue_name)


def _distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt(sum(((a[i] - b[i]) ** 2 for i in range(3))))


def _read_pdb_residues(pdb_file: Path) -> List[Dict[str, object]]:
    residues: List[Dict[str, object]] = []
    current_key = None
    current = None
    for raw in pdb_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not raw.startswith("ATOM"):
            continue
        line = raw.ljust(80)
        key = _pdb_residue_key(line)
        if key != current_key:
            current = {
                "original_chain": key[0] or "A",
                "residue_number": key[1],
                "insertion_code": key[2],
                "residue_name": key[3],
                "lines": [],
                "atoms": {},
            }
            residues.append(current)
            current_key = key
        assert current is not None
        current["lines"].append(line)
        atom_name = line[12:16].strip().upper()
        try:
            coord = parse_pdb_atom_coord(line)
        except Exception:
            continue
        current["atoms"][atom_name] = coord
    return residues


def _read_ligand_heavy_coords(ligand_pdb: Path) -> List[Tuple[float, float, float]]:
    coords: List[Tuple[float, float, float]] = []
    for raw in ligand_pdb.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not raw.startswith(("ATOM", "HETATM")):
            continue
        line = raw.ljust(80)
        if _pdb_element_from_line(line) == "H":
            continue
        try:
            coords.append(parse_pdb_atom_coord(line))
        except Exception:
            pass
    return coords


def _residue_min_distance_to_ligand(
    residue: Dict[str, object], ligand_coords: List[Tuple[float, float, float]]
) -> Optional[float]:
    if not ligand_coords:
        return None
    residue_coords = list(residue.get("atoms", {}).values())
    if not residue_coords:
        return None
    return min((_distance(a, b) for a in residue_coords for b in ligand_coords))


def _chain_id_pool() -> List[str]:
    return list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")


def split_receptor_at_coordinate_gaps(
    input_pdb: Path,
    output_pdb: Path,
    native_ligand_pdb: Path,
    report_json: Path,
    force: bool,
) -> Dict[str, object]:
    if (
        output_pdb.exists()
        and output_pdb.stat().st_size > 0
        and report_json.exists()
        and (not force)
    ):
        try:
            cached = json.loads(report_json.read_text(encoding="utf-8"))
        except Exception:
            cached = {}
        print(f"[RESUME] Reusing the existing chain-break output: {output_pdb}")
        return cached
    residues = _read_pdb_residues(input_pdb)
    if not residues:
        raise RuntimeError(
            f"Could not read ATOM residues from receptor PDB: {input_pdb}"
        )
    ligand_coords = _read_ligand_heavy_coords(native_ligand_pdb)
    pool = _chain_id_pool()
    segments: List[List[Dict[str, object]]] = []
    breaks: List[Dict[str, object]] = []
    current_segment: List[Dict[str, object]] = []
    for idx, residue in enumerate(residues):
        should_break = False
        reasons: List[str] = []
        cn_distance: Optional[float] = None
        if idx > 0:
            previous = residues[idx - 1]
            if residue["original_chain"] != previous["original_chain"]:
                should_break = True
                reasons.append("source chain changed")
            else:
                prev_num = int(previous["residue_number"])
                curr_num = int(residue["residue_number"])
                prev_icode = str(previous["insertion_code"])
                curr_icode = str(residue["insertion_code"])
                if not prev_icode and (not curr_icode) and (curr_num - prev_num > 1):
                    should_break = True
                    reasons.append(f"residue numbering gap {prev_num}->{curr_num}")
                elif curr_num < prev_num:
                    should_break = True
                    reasons.append(f"residue numbering reversal {prev_num}->{curr_num}")
                prev_c = previous.get("atoms", {}).get("C")
                curr_n = residue.get("atoms", {}).get("N")
                if prev_c is None or curr_n is None:
                    should_break = True
                    reasons.append("backbone C or N is missing at the break")
                else:
                    cn_distance = _distance(prev_c, curr_n)
                    if cn_distance > PEPTIDE_BOND_MAX_DISTANCE:
                        should_break = True
                        reasons.append(
                            f"backbone C-N distance {cn_distance:.3f}Å>{PEPTIDE_BOND_MAX_DISTANCE:.2f}Å"
                        )
            if should_break:
                prev_lig_dist = _residue_min_distance_to_ligand(previous, ligand_coords)
                curr_lig_dist = _residue_min_distance_to_ligand(residue, ligand_coords)
                values = [x for x in (prev_lig_dist, curr_lig_dist) if x is not None]
                min_lig_dist = min(values) if values else None
                breaks.append(
                    {
                        "previous": {
                            "chain": previous["original_chain"],
                            "residue_number": previous["residue_number"],
                            "insertion_code": previous["insertion_code"],
                            "residue_name": previous["residue_name"],
                        },
                        "next": {
                            "chain": residue["original_chain"],
                            "residue_number": residue["residue_number"],
                            "insertion_code": residue["insertion_code"],
                            "residue_name": residue["residue_name"],
                        },
                        "reasons": reasons,
                        "backbone_C_N_distance_A": cn_distance,
                        "min_boundary_distance_to_native_ligand_A": min_lig_dist,
                        "near_binding_pocket": min_lig_dist is not None
                        and min_lig_dist < CHAIN_BREAK_POCKET_WARNING_DISTANCE,
                    }
                )
        if should_break and current_segment:
            segments.append(current_segment)
            current_segment = []
        current_segment.append(residue)
    if current_segment:
        segments.append(current_segment)
    if len(segments) > len(pool):
        raise RuntimeError(
            f"Detected {len(segments)} coordinate segments, exceeding the capacity of single-character PDB chain IDs."
        )
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pdb, "w", encoding="utf-8") as handle:
        handle.write("REMARK 9GU4 coordinate segments split at unresolved chain gaps\n")
        handle.write(
            f"REMARK peptide C-N threshold {PEPTIDE_BOND_MAX_DISTANCE:.2f} Angstrom\n"
        )
        serial = 1
        segment_report: List[Dict[str, object]] = []
        for segment_index, segment in enumerate(segments):
            new_chain = pool[segment_index]
            segment_report.append(
                {
                    "segment_index": segment_index + 1,
                    "new_chain_id": new_chain,
                    "original_chain_id": segment[0]["original_chain"],
                    "first_residue": f"{segment[0]['residue_name']} {segment[0]['residue_number']}{segment[0]['insertion_code']}",
                    "last_residue": f"{segment[-1]['residue_name']} {segment[-1]['residue_number']}{segment[-1]['insertion_code']}",
                    "residue_count": len(segment),
                }
            )
            for residue in segment:
                for source_line in residue["lines"]:
                    line = source_line.ljust(80)
                    rebuilt = (
                        line[:6] + f"{serial:5d}" + line[11:21] + new_chain + line[22:]
                    )
                    handle.write(rebuilt.rstrip() + "\n")
                    serial += 1
            handle.write("TER\n")
        handle.write("END\n")
    near_breaks = [x for x in breaks if x["near_binding_pocket"]]
    report: Dict[str, object] = {
        "created_at": now_text(),
        "repair_pipeline_version": REPAIR_PIPELINE_VERSION,
        "input_pdb": str(input_pdb),
        "output_pdb": str(output_pdb),
        "enabled": True,
        "peptide_bond_max_distance_A": PEPTIDE_BOND_MAX_DISTANCE,
        "pocket_warning_distance_A": CHAIN_BREAK_POCKET_WARNING_DISTANCE,
        "input_residue_count": len(residues),
        "segment_count": len(segments),
        "break_count": len(breaks),
        "near_pocket_break_count": len(near_breaks),
        "segments": segment_report,
        "breaks": breaks,
    }
    atomic_write_json(report, report_json)
    print("\n" + "=" * 90)
    print("[TOPOLOGY] Inspecting coordinate-chain breaks in 9GU4")
    print(
        f"[TOPOLOGY] Input residues={len(residues)}, segments={len(segments)}, breaks={len(breaks)}"
    )
    print(f"[OK] Segmented receptor PDB: {output_pdb}")
    print(f"[OK] Chain-break report: {report_json}")
    for item in breaks:
        prev = item["previous"]
        nxt = item["next"]
        distance_text = item["min_boundary_distance_to_native_ligand_A"]
        distance_text = "NA" if distance_text is None else f"{distance_text:.2f}Å"
        print(
            f"[BREAK] {prev['chain']}:{prev['residue_number']}{prev['insertion_code']} -> {nxt['chain']}:{nxt['residue_number']}{nxt['insertion_code']} | {'; '.join(item['reasons'])} | distance to NP3-253={distance_text}"
        )
    if near_breaks:
        message = f"Found {len(near_breaks)} chain-break endpoints within {CHAIN_BREAK_POCKET_WARNING_DISTANCE:.1f} A of NP3-253. Inspect chain_break_report.json and the 3D structure before interpreting docking results."
        if FAIL_IF_CHAIN_BREAK_NEAR_POCKET:
            raise RuntimeError(message)
        print(f"[WARN] {message}")
    print("=" * 90)
    return report


def _pdbfixer_missing_residues_report(fixer) -> List[Dict[str, object]]:
    report: List[Dict[str, object]] = []
    chains = list(fixer.topology.chains())
    missing = getattr(fixer, "missingResidues", {}) or {}
    for key, residue_names in missing.items():
        try:
            chain_index, insertion_index = key
            chain_id = (
                chains[chain_index].id
                if chain_index < len(chains)
                else str(chain_index)
            )
        except Exception:
            chain_id = ""
            insertion_index = None
        report.append(
            {
                "chain_id": chain_id,
                "insert_before_residue_index": insertion_index,
                "residue_names": list(residue_names),
                "count": len(residue_names),
            }
        )
    return report


def _pdbfixer_missing_atoms_report(fixer) -> List[Dict[str, object]]:
    by_residue: Dict[Tuple[str, str, str, str], set] = {}

    def residue_key(residue) -> Tuple[str, str, str, str]:
        chain = getattr(residue, "chain", None)
        chain_id = getattr(chain, "id", "") if chain is not None else ""
        return (
            str(chain_id),
            str(getattr(residue, "id", "")),
            str(getattr(residue, "insertionCode", "") or ""),
            str(getattr(residue, "name", "")),
        )

    for residue, atoms in (getattr(fixer, "missingAtoms", {}) or {}).items():
        key = residue_key(residue)
        names = by_residue.setdefault(key, set())
        for atom in atoms:
            names.add(str(getattr(atom, "name", atom)))
    for residue, atoms in (getattr(fixer, "missingTerminals", {}) or {}).items():
        key = residue_key(residue)
        names = by_residue.setdefault(key, set())
        for atom in atoms:
            names.add(str(getattr(atom, "name", atom)))
    report: List[Dict[str, object]] = []
    for (chain_id, residue_id, insertion_code, residue_name), atom_names in sorted(
        by_residue.items()
    ):
        report.append(
            {
                "chain_id": chain_id,
                "residue_id": residue_id,
                "insertion_code": insertion_code,
                "residue_name": residue_name,
                "missing_atom_names": sorted(atom_names),
                "missing_atom_count": len(atom_names),
            }
        )
    return report


def compute_box_from_native_ligand(
    ligand_pdb: Path,
    padding: float,
    min_size: Tuple[float, float, float],
    max_size: Tuple[float, float, float],
) -> Tuple[
    Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]
]:
    coords: List[Tuple[float, float, float]] = []
    for line in ligand_pdb.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            coords.append(parse_pdb_atom_coord(line))
    if not coords:
        raise RuntimeError(
            f"Could not parse coordinates from the native ligand: {ligand_pdb}"
        )
    mins = tuple((min((c[i] for c in coords)) for i in range(3)))
    maxs = tuple((max((c[i] for c in coords)) for i in range(3)))
    center = tuple(((mins[i] + maxs[i]) / 2.0 for i in range(3)))
    span = tuple((maxs[i] - mins[i] for i in range(3)))
    sizes = []
    for i in range(3):
        raw = span[i] + 2.0 * padding
        size = max(min_size[i], raw)
        if size > max_size[i]:
            print(
                f"[WARN] Automatically derived box axis {i + 1} requires {size:.2f} A, exceeding the maximum {max_size[i]:.2f} A; the size was clipped."
            )
            size = max_size[i]
        sizes.append(size)
    box_size = tuple(sizes)
    print("[BOX] native ligand span =", tuple((round(x, 3) for x in span)))
    print("[BOX] center =", tuple((round(x, 3) for x in center)))
    print("[BOX] size =", tuple((round(x, 3) for x in box_size)))
    return (center, box_size, span)


def write_vina_config(
    config_file: Path,
    center: Tuple[float, float, float],
    box_size: Tuple[float, float, float],
) -> None:
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_text = f"# Auto-generated Vina config for NLRP3 {PDB_ID}\ncenter_x = {center[0]:.3f}\ncenter_y = {center[1]:.3f}\ncenter_z = {center[2]:.3f}\n\nsize_x = {box_size[0]:.3f}\nsize_y = {box_size[1]:.3f}\nsize_z = {box_size[2]:.3f}\n\nexhaustiveness = {EXHAUSTIVENESS}\nnum_modes = {NUM_MODES}\nenergy_range = {ENERGY_RANGE}\ncpu = {CPU}\nseed = {SEED}\n"
    config_file.write_text(config_text, encoding="utf-8")
    print(f"[OK] Vina config：{config_file}")


def _parse_pdb_residue_coordinates(
    pdb_file: Path,
) -> Dict[Tuple[str, str, str, str], List[Tuple[float, float, float]]]:
    residues: Dict[Tuple[str, str, str, str], List[Tuple[float, float, float]]] = {}
    for line in pdb_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if len(line) < 54:
            continue
        chain_id = line[21].strip()
        residue_id = line[22:26].strip()
        insertion_code = line[26].strip()
        residue_name = line[17:20].strip()
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        residues.setdefault(
            (chain_id, residue_id, insertion_code, residue_name), []
        ).append(xyz)
    return residues


def _minimum_distance_between_coordinate_sets(
    first: Sequence[Tuple[float, float, float]],
    second: Sequence[Tuple[float, float, float]],
) -> Optional[float]:
    if not first or not second:
        return None
    best = float("inf")
    for x1, y1, z1 in first:
        for x2, y2, z2 in second:
            d2 = (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2
            if d2 < best:
                best = d2
    return math.sqrt(best)


def audit_receptor_missing_atoms(
    receptor_pdb: Path, native_ligand_pdb: Path, report_json: Path, force: bool
) -> Dict[str, object]:
    if report_json.exists() and (not force):
        try:
            cached = json.loads(report_json.read_text(encoding="utf-8"))
            if cached.get("audit_version") == "9gu4-adt-audit-v1":
                print(
                    f"[RESUME] Reusing the existing receptor-integrity audit: {report_json}"
                )
                return cached
        except Exception:
            pass
    report: Dict[str, object] = {
        "audit_version": "9gu4-adt-audit-v1",
        "created_at": now_text(),
        "receptor_pdb": str(receptor_pdb),
        "native_ligand_pdb": str(native_ligand_pdb),
        "structure_modified": False,
        "pdbfixer_available": False,
        "warning_distance_angstrom": POCKET_INCOMPLETE_RESIDUE_WARNING_DISTANCE,
    }
    try:
        from pdbfixer import PDBFixer
    except ImportError:
        report.update(
            {
                "status": "skipped_missing_pdbfixer",
                "message": "PDBFixer is unavailable, so the missing-side-chain audit was not run; AutoDockTools can still prepare the receptor.",
            }
        )
        atomic_write_json(report, report_json)
        print(
            "[WARN] PDBFixer is unavailable; skipping the receptor missing-atom audit."
        )
        return report
    report["pdbfixer_available"] = True
    fixer = PDBFixer(filename=str(receptor_pdb))
    fixer.findMissingResidues()
    whole_missing = _pdbfixer_missing_residues_report(fixer)
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    missing_atoms = _pdbfixer_missing_atoms_report(fixer)
    residue_coords = _parse_pdb_residue_coordinates(receptor_pdb)
    ligand_coords_by_residue = _parse_pdb_residue_coordinates(native_ligand_pdb)
    ligand_coords = [
        xyz for coordinates in ligand_coords_by_residue.values() for xyz in coordinates
    ]
    enriched: List[Dict[str, object]] = []
    for item in missing_atoms:
        key = (
            str(item.get("chain_id", "")),
            str(item.get("residue_id", "")),
            str(item.get("insertion_code", "")),
            str(item.get("residue_name", "")),
        )
        coordinates = residue_coords.get(key, [])
        if not coordinates:
            for candidate_key, candidate_coords in residue_coords.items():
                if candidate_key[:3] == key[:3]:
                    coordinates = candidate_coords
                    break
        distance = _minimum_distance_between_coordinate_sets(coordinates, ligand_coords)
        enriched_item = dict(item)
        enriched_item["minimum_distance_to_native_ligand_angstrom"] = (
            round(distance, 3) if distance is not None else None
        )
        enriched_item["near_binding_pocket"] = (
            distance is not None
            and distance <= POCKET_INCOMPLETE_RESIDUE_WARNING_DISTANCE
        )
        enriched.append(enriched_item)
    enriched.sort(
        key=lambda x: (
            x["minimum_distance_to_native_ligand_angstrom"] is None,
            (
                x["minimum_distance_to_native_ligand_angstrom"]
                if x["minimum_distance_to_native_ligand_angstrom"] is not None
                else float("inf")
            ),
        )
    )
    near_pocket = [x for x in enriched if x.get("near_binding_pocket")]
    missing_atom_count = sum((int(x.get("missing_atom_count", 0)) for x in enriched))
    whole_missing_count = sum((int(x.get("count", 0)) for x in whole_missing))
    report.update(
        {
            "status": "completed",
            "whole_missing_residue_segments": whole_missing,
            "whole_missing_residue_count": whole_missing_count,
            "incomplete_modeled_residues": enriched,
            "incomplete_modeled_residue_count": len(enriched),
            "missing_heavy_or_terminal_atom_count": missing_atom_count,
            "near_pocket_incomplete_residues": near_pocket,
            "near_pocket_incomplete_residue_count": len(near_pocket),
            "passed_pocket_check": len(near_pocket) == 0,
        }
    )
    atomic_write_json(report, report_json)
    print("\n" + "=" * 90)
    print("[AUDIT] Read-only integrity check of the experimental 9GU4 receptor")
    print(f"[AUDIT] Entirely unresolved residues={whole_missing_count}")
    print(
        f"[AUDIT] Incomplete modeled residues={len(enriched)}, missing heavy/terminal atoms={missing_atom_count}"
    )
    print(
        f"[AUDIT] Incomplete residues within {POCKET_INCOMPLETE_RESIDUE_WARNING_DISTANCE:.1f} A of NP3-253={len(near_pocket)}"
    )
    for item in near_pocket[:20]:
        print(
            f"[POCKET-WARN] {item.get('chain_id')}:{item.get('residue_name')}{item.get('residue_id')}{item.get('insertion_code') or ''} | missing={','.join(item.get('missing_atom_names', []))} | distance={item.get('minimum_distance_to_native_ligand_angstrom')} Å"
        )
    print(f"[AUDIT] Full report: {report_json}")
    print("=" * 90)
    if near_pocket and FAIL_IF_INCOMPLETE_RESIDUE_NEAR_POCKET:
        raise RuntimeError(
            f"Incomplete residues were detected near the NP3-253 pocket; stopping as configured.\nReport: {report_json}"
        )
    return report


def _conda_environment_exists(environment_name: str) -> bool:
    conda = shutil.which("conda")
    if conda is None or not environment_name:
        return False
    try:
        process = subprocess.run(
            [conda, "env", "list", "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
            check=False,
        )
        if process.returncode != 0:
            return False
        data = json.loads(process.stdout)
        for prefix in data.get("envs", []):
            if Path(prefix).name == environment_name:
                return True
    except Exception:
        return False
    return False


def _probe_prepare_receptor4_command(prefix: Sequence[str]) -> bool:
    try:
        process = subprocess.run(
            [*map(str, prefix), "-h"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=45,
            check=False,
        )
    except Exception:
        return False
    output = (process.stdout or "").lower()
    return (
        "prepare_receptor" in output
        or "receptor_filename" in output
        or "usage:" in output
    )


def resolve_prepare_receptor4_command(
    explicit: Optional[str], conda_environment: Optional[str]
) -> Tuple[List[str], Dict[str, object]]:
    candidates: List[Tuple[List[str], str]] = []
    configured = explicit or AUTODOCKTOOLS_EXPLICIT_PREPARE_RECEPTOR4
    if configured:
        path = Path(configured).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"The specified prepare_receptor4 executable does not exist: {path}"
            )
        candidates.append(([str(path)], "explicit_path"))
    for executable in ("prepare_receptor4.py", "prepare_receptor4"):
        located = shutil.which(executable)
        if located:
            candidates.append(([located], "current_path"))
    env_name = conda_environment or AUTODOCKTOOLS_CONDA_ENV
    conda = shutil.which("conda")
    if conda and env_name and _conda_environment_exists(env_name):
        candidates.append(
            (
                [
                    conda,
                    "run",
                    "--no-capture-output",
                    "-n",
                    env_name,
                    "prepare_receptor4.py",
                ],
                f"conda_env:{env_name}",
            )
        )
        candidates.append(
            (
                [
                    conda,
                    "run",
                    "--no-capture-output",
                    "-n",
                    env_name,
                    "prepare_receptor4",
                ],
                f"conda_env:{env_name}",
            )
        )
    home = Path.home()
    common_scripts = [
        home
        / "MGLTools-1.5.6"
        / "MGLToolsPckgs"
        / "AutoDockTools"
        / "Utilities24"
        / "prepare_receptor4.py",
        home
        / "MGLTools-1.5.7"
        / "MGLToolsPckgs"
        / "AutoDockTools"
        / "Utilities24"
        / "prepare_receptor4.py",
        Path(
            "/opt/mgltools/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py"
        ),
        Path(
            "/usr/local/MGLTools/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py"
        ),
    ]
    for script in common_scripts:
        if not script.exists():
            continue
        pythonsh_candidates = [
            script.parents[3] / "bin" / "pythonsh",
            script.parents[3] / "bin" / "python",
            shutil.which("pythonsh"),
        ]
        for pythonsh in pythonsh_candidates:
            if pythonsh and Path(str(pythonsh)).exists():
                candidates.append(([str(pythonsh), str(script)], "mgltools_pythonsh"))
    attempted: List[str] = []
    for prefix, source in candidates:
        signature = " ".join(prefix)
        if signature in attempted:
            continue
        attempted.append(signature)
        if _probe_prepare_receptor4_command(prefix):
            info = {
                "source": source,
                "command_prefix": prefix,
                "conda_environment": (
                    env_name if source.startswith("conda_env:") else None
                ),
            }
            print(f"[OK] AutoDockTools prepare_receptor4：{signature}")
            return (prefix, info)
    raise RuntimeError(
        f"AutoDockTools prepare_receptor4.py was not found.\n\nInstall it in a separate environment:\n  conda create -n mgltools -c bioconda mgltools=1.5.7\n\nThe script will call it as:\n  conda run -n mgltools prepare_receptor4.py ...\n\nAlternatively specify it explicitly:\n  python dock_nlrp3_candidates.py --prepare-receptor4 /path/to/prepare_receptor4.py\nTried: {attempted}"
    )


def validate_receptor_pdbqt(
    receptor_pdbqt: Path, report_json: Path
) -> Dict[str, object]:
    if not receptor_pdbqt.exists() or receptor_pdbqt.stat().st_size == 0:
        raise RuntimeError(f"Receptor PDBQT is missing or empty: {receptor_pdbqt}")
    atom_count = 0
    hydrogen_count = 0
    invalid_lines: List[Dict[str, object]] = []
    residue_names: Set[str] = set()
    atom_types: Set[str] = set()
    for line_number, line in enumerate(
        receptor_pdbqt.read_text(encoding="utf-8", errors="ignore").splitlines(),
        start=1,
    ):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        atom_count += 1
        residue_names.add(line[17:20].strip().upper())
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            if not all((math.isfinite(v) for v in (x, y, z))):
                raise ValueError("non-finite coordinate")
        except Exception as exc:
            invalid_lines.append(
                {
                    "line_number": line_number,
                    "reason": f"coordinate: {exc}",
                    "line": line,
                }
            )
            continue
        fields = line.split()
        if len(fields) < 2:
            invalid_lines.append(
                {"line_number": line_number, "reason": "too_few_fields", "line": line}
            )
            continue
        atom_type = fields[-1]
        atom_types.add(atom_type)
        try:
            charge = float(fields[-2])
            if not math.isfinite(charge):
                raise ValueError("non-finite charge")
        except Exception as exc:
            invalid_lines.append(
                {"line_number": line_number, "reason": f"charge: {exc}", "line": line}
            )
        if atom_type.upper() in {"H", "HD", "HS"}:
            hydrogen_count += 1
    unwanted = sorted(
        (
            name
            for name in residue_names
            if name in {"HOH", "WAT", "NP3", "ADP", "MG", "NA", "CL"}
        )
    )
    passed = atom_count > 0 and (not invalid_lines) and (not unwanted)
    report = {
        "created_at": now_text(),
        "receptor_pdbqt": str(receptor_pdbqt),
        "atom_count": atom_count,
        "polar_hydrogen_like_atom_count": hydrogen_count,
        "atom_types": sorted(atom_types),
        "residue_names": sorted(residue_names),
        "unwanted_residue_names": unwanted,
        "invalid_line_count": len(invalid_lines),
        "invalid_lines_preview": invalid_lines[:30],
        "validation_passed": passed,
    }
    atomic_write_json(report, report_json)
    if not passed:
        raise RuntimeError(
            f"The receptor PDBQT generated by AutoDockTools failed validation.\nReport: {report_json}"
        )
    print(
        f"[OK] Receptor PDBQT validation passed: atoms={atom_count}, polar-H-like={hydrogen_count}, atom_types={len(atom_types)}"
    )
    print(f"[OK] PDBQT validation report: {report_json}")
    return report


def prepare_receptor(
    receptor_pdb: Path,
    receptor_dir: Path,
    target_name: str,
    center: Tuple[float, float, float],
    box_size: Tuple[float, float, float],
    force: bool,
    explicit_prepare_receptor4: Optional[str] = None,
    autodocktools_conda_env: Optional[str] = None,
) -> Tuple[Path, Dict[str, object]]:
    del center, box_size
    receptor_dir.mkdir(parents=True, exist_ok=True)
    output_pdbqt = receptor_dir / f"{target_name}.pdbqt"
    log_file = receptor_dir / "prepare_receptor4.log"
    validation_json = receptor_dir / "receptor_pdbqt_validation.json"
    metadata_json = receptor_dir / "autodocktools_receptor_prep.json"
    if output_pdbqt.exists() and output_pdbqt.stat().st_size > 0 and (not force):
        validation = validate_receptor_pdbqt(output_pdbqt, validation_json)
        metadata = {}
        if metadata_json.exists():
            try:
                metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
            except Exception:
                metadata = {}
        metadata["validation"] = validation
        print(
            f"[RESUME] Reusing the existing AutoDockTools receptor PDBQT: {output_pdbqt}"
        )
        return (output_pdbqt, metadata)
    prefix, command_info = resolve_prepare_receptor4_command(
        explicit=explicit_prepare_receptor4, conda_environment=autodocktools_conda_env
    )
    command = [
        *prefix,
        "-r",
        str(receptor_pdb),
        "-o",
        str(output_pdbqt),
        "-A",
        AUTODOCKTOOLS_REPAIRS,
        "-U",
        AUTODOCKTOOLS_CLEANUP,
        "-v",
    ]
    print(
        "[RECEPTOR] Preparing 9GU4 with the AutoDockTools workflow used in the cited study."
    )
    print(
        f"[RECEPTOR] repairs={AUTODOCKTOOLS_REPAIRS}, cleanup={AUTODOCKTOOLS_CLEANUP}, charges=Gasteiger(default)"
    )
    run_cmd(command, log_file=log_file)
    validation = validate_receptor_pdbqt(output_pdbqt, validation_json)
    metadata = {
        "created_at": now_text(),
        "backend": "AutoDockTools prepare_receptor4.py",
        "input_receptor_pdb": str(receptor_pdb),
        "output_receptor_pdbqt": str(output_pdbqt),
        "command": command,
        "command_info": command_info,
        "repairs": AUTODOCKTOOLS_REPAIRS,
        "cleanup": AUTODOCKTOOLS_CLEANUP,
        "charges": "Gasteiger (AutoDockTools default)",
        "nonpolar_hydrogens": "merged by nphs cleanup",
        "log_file": str(log_file),
        "validation_report": str(validation_json),
        "validation": validation,
    }
    atomic_write_json(metadata, metadata_json)
    print(f"[OK] AutoDockTools receptor PDBQT：{output_pdbqt}")
    return (output_pdbqt, metadata)


def generate_ligand_3d_sdf(
    record: LigandRecord, smi_file: Path, sdf_file: Path, log_file: Path, force: bool
) -> None:
    if sdf_file.exists() and sdf_file.stat().st_size > 0 and (not force):
        print(f"[RESUME] Reusing existing 3D SDF: {sdf_file}")
        return
    smi_file.parent.mkdir(parents=True, exist_ok=True)
    sdf_file.parent.mkdir(parents=True, exist_ok=True)
    smi_file.write_text(f"{record.smiles}\t{record.safe_name}\n", encoding="utf-8")
    command = [
        "obabel",
        "-ismi",
        str(smi_file),
        "-osdf",
        "-O",
        str(sdf_file),
        "--gen3d",
        "-p",
        str(LIGAND_PH),
        "-h",
    ]
    run_cmd(command, log_file=log_file)
    if not sdf_file.exists() or sdf_file.stat().st_size == 0:
        raise RuntimeError(f"Open Babel did not generate a valid SDF: {sdf_file}")


def prepare_ligand_pdbqt(
    sdf_file: Path, pdbqt_file: Path, log_file: Path, force: bool
) -> None:
    if pdbqt_file.exists() and pdbqt_file.stat().st_size > 0 and (not force):
        print(f"[RESUME] Reusing existing ligand PDBQT: {pdbqt_file}")
        return
    pdbqt_file.parent.mkdir(parents=True, exist_ok=True)
    command = ["mk_prepare_ligand.py", "-i", str(sdf_file), "-o", str(pdbqt_file)]
    run_cmd(command, log_file=log_file)
    if not pdbqt_file.exists() or pdbqt_file.stat().st_size == 0:
        raise RuntimeError(f"Meeko did not generate a valid ligand PDBQT: {pdbqt_file}")


def parse_vina_score_from_log(log_file: Path) -> Optional[float]:
    if not log_file.exists():
        return None
    for line in log_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = re.match("^\\s*1\\s+(-?\\d+(?:\\.\\d+)?)\\s+", line)
        if match:
            return float(match.group(1))
    return None


def parse_vina_score_from_pdbqt(pdbqt_file: Path) -> Optional[float]:
    if not pdbqt_file.exists():
        return None
    for line in pdbqt_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = re.search("REMARK\\s+VINA\\s+RESULT:\\s*(-?\\d+(?:\\.\\d+)?)", line)
        if match:
            return float(match.group(1))
    return None


def run_vina_docking(
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    config_file: Path,
    out_pdbqt: Path,
    log_file: Path,
    force: bool,
) -> float:
    if out_pdbqt.exists() and out_pdbqt.stat().st_size > 0 and (not force):
        score = parse_vina_score_from_log(log_file)
        if score is None:
            score = parse_vina_score_from_pdbqt(out_pdbqt)
        if score is not None:
            print(f"[RESUME] Reusing existing Vina result: {out_pdbqt}，score={score}")
            return score
    out_pdbqt.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "vina",
        "--receptor",
        str(receptor_pdbqt),
        "--ligand",
        str(ligand_pdbqt),
        "--config",
        str(config_file),
        "--out",
        str(out_pdbqt),
    ]
    run_cmd(command, log_file=log_file, timeout_seconds=VINA_TIMEOUT_SECONDS)
    if not out_pdbqt.exists() or out_pdbqt.stat().st_size == 0:
        raise RuntimeError(f"Vina output is empty: {out_pdbqt}")
    score = parse_vina_score_from_log(log_file)
    if score is None:
        score = parse_vina_score_from_pdbqt(out_pdbqt)
    if score is None or not math.isfinite(score):
        raise RuntimeError(
            f"Vina completed, but the mode-1 score could not be parsed: {log_file} / {out_pdbqt}"
        )
    return score


def extract_first_vina_model(input_pdbqt: Path, output_pdbqt: Path) -> None:
    lines = input_pdbqt.read_text(encoding="utf-8", errors="ignore").splitlines(True)
    has_models = any((line.startswith("MODEL") for line in lines))
    if not has_models:
        output_pdbqt.write_text("".join(lines), encoding="utf-8")
        return
    selected: List[str] = []
    inside_first = False
    for line in lines:
        if line.startswith("MODEL"):
            if inside_first:
                break
            inside_first = True
            selected.append(line)
            continue
        if inside_first:
            selected.append(line)
            if line.startswith("ENDMDL"):
                break
    output_pdbqt.write_text("".join(selected), encoding="utf-8")


def convert_best_pose(
    all_poses_pdbqt: Path,
    best_pose_pdbqt: Path,
    best_pose_sdf: Path,
    best_pose_pdb: Path,
    conversion_log: Path,
) -> None:
    extract_first_vina_model(all_poses_pdbqt, best_pose_pdbqt)
    run_cmd(
        ["obabel", "-ipdbqt", str(best_pose_pdbqt), "-osdf", "-O", str(best_pose_sdf)],
        log_file=conversion_log.with_name(conversion_log.stem + "_sdf.log"),
    )
    run_cmd(
        ["obabel", "-ipdbqt", str(best_pose_pdbqt), "-opdb", "-O", str(best_pose_pdb)],
        log_file=conversion_log.with_name(conversion_log.stem + "_pdb.log"),
    )


def combine_receptor_and_ligand_pdb(
    receptor_pdb: Path, ligand_pdb: Path, complex_pdb: Path
) -> None:
    receptor_lines: List[str] = []
    max_receptor_serial = 0
    for line in receptor_pdb.read_text(encoding="utf-8", errors="ignore").splitlines(
        True
    ):
        if line.startswith(("ATOM", "TER")):
            receptor_lines.append(line)
        if line.startswith("ATOM"):
            try:
                max_receptor_serial = max(max_receptor_serial, int(line[6:11]))
            except ValueError:
                pass
    source_lines = ligand_pdb.read_text(encoding="utf-8", errors="ignore").splitlines(
        True
    )
    atom_lines = [line for line in source_lines if line.startswith(("ATOM", "HETATM"))]
    if not atom_lines:
        raise RuntimeError(
            f"Could not extract atoms from the best-pose ligand PDB: {ligand_pdb}"
        )
    serial_map: Dict[int, int] = {}
    ligand_lines: List[str] = []
    next_serial = max_receptor_serial + 1
    for line in atom_lines:
        try:
            old_serial = int(line[6:11])
        except ValueError:
            old_serial = next_serial
        new_serial = next_serial
        next_serial += 1
        serial_map[old_serial] = new_serial
        padded = line.rstrip("\n").ljust(80)
        rebuilt = (
            "HETATM"
            + f"{new_serial:5d}"
            + padded[11:17]
            + "LIG"
            + " Z"
            + f"{1:4d}"
            + padded[26:]
        )
        ligand_lines.append(rebuilt.rstrip() + "\n")
    conect_lines: List[str] = []
    for line in source_lines:
        if not line.startswith("CONECT"):
            continue
        try:
            old_numbers = [int(x) for x in line[6:].split()]
        except ValueError:
            continue
        new_numbers = [serial_map[x] for x in old_numbers if x in serial_map]
        if len(new_numbers) >= 2:
            conect_lines.append(
                "CONECT" + "".join((f"{x:5d}" for x in new_numbers)) + "\n"
            )
    complex_pdb.parent.mkdir(parents=True, exist_ok=True)
    with open(complex_pdb, "w", encoding="utf-8") as f:
        f.write("REMARK NLRP3 9GU4 receptor + Vina best ligand pose\n")
        f.writelines(receptor_lines)
        f.writelines(ligand_lines)
        f.writelines(conect_lines)
        f.write("END\n")


def write_pymol_script(
    receptor_pdb: Path, ligand_pdb: Path, output_pml: Path, compound_name: str
) -> None:
    script = f"# Auto-generated PyMOL script for {compound_name}\nreinitialize\nload {receptor_pdb.as_posix()}, receptor\nload {ligand_pdb.as_posix()}, ligand\nhide everything\nshow cartoon, receptor\ncolor gray80, receptor\nshow sticks, ligand\ncolor cyan, ligand\nselect pocket, byres (receptor within 5.0 of ligand)\nshow sticks, pocket\ncolor wheat, pocket\nset stick_radius, 0.18\nset cartoon_transparency, 0.10\nset bg_rgb, [1, 1, 1]\nset ray_opaque_background, off\nzoom ligand, 12\n"
    output_pml.write_text(script, encoding="utf-8")


RESULT_FIELDS = [
    "input_row",
    "dta_rank",
    "chembl_id",
    "compound_name",
    "file_safe_name",
    "smiles",
    "vina_score_kcal_mol",
    "status",
    "started_at",
    "finished_at",
    "error",
    "warning",
    "ligand_smi",
    "ligand_sdf",
    "ligand_pdbqt",
    "vina_all_poses_pdbqt",
    "vina_log",
    "best_pose_pdbqt",
    "best_pose_sdf",
    "best_pose_pdb",
    "complex_pdb",
    "pymol_script",
    "status_json",
]


def write_results_csv(rows: List[Dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(output)


def update_summary_files(
    result_rows: List[Dict[str, object]], results_dir: Path
) -> Tuple[Path, Path]:
    original_csv = results_dir / f"{TARGET_NAME}_vina_results_all.csv"
    sorted_csv = results_dir / f"{TARGET_NAME}_vina_scores_sorted.csv"
    write_results_csv(result_rows, original_csv)
    successful = [
        row
        for row in result_rows
        if row.get("status") == "success"
        and isinstance(row.get("vina_score_kcal_mol"), (int, float))
    ]
    successful_sorted = sorted(
        successful, key=lambda row: float(row["vina_score_kcal_mol"])
    )
    write_results_csv(successful_sorted, sorted_csv)
    return (original_csv, sorted_csv)


def load_existing_status(status_json: Path) -> Optional[Dict[str, object]]:
    if not status_json.exists():
        return None
    try:
        return json.loads(status_json.read_text(encoding="utf-8"))
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AutoDock Vina docking of nicergoline, nalfurafine, and revumenib against NLRP3 structure 9GU4."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data" / "docking_results" / "NLRP3_candidates",
        help="Output root directory.",
    )
    parser.add_argument(
        "--structure-file",
        "--pdb-file",
        dest="structure_file",
        type=Path,
        default=None,
        help="Optional local 9GU4 mmCIF file (.cif or .mmcif); --pdb-file is retained as a compatibility alias.",
    )
    parser.add_argument(
        "--min-rank",
        type=int,
        default=None,
        help="Only process compounds whose original rank is at least this value.",
    )
    parser.add_argument(
        "--max-rank",
        type=int,
        default=None,
        help="Only process compounds whose original rank is at most this value.",
    )
    parser.add_argument(
        "--chembl-id",
        action="append",
        default=[],
        help="Only process the specified ChEMBL ID; repeat this option for multiple IDs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the receptor and ligands and rerun completed docking jobs.",
    )
    parser.add_argument(
        "--skip-receptor-repair",
        action="store_true",
        help="Compatibility option; the current workflow does not modify the receptor with PDBFixer.",
    )
    parser.add_argument(
        "--skip-chain-segmentation",
        action="store_true",
        help="Skip the coordinate-break audit and segmented preview.",
    )
    parser.add_argument(
        "--skip-minimization",
        action="store_true",
        help="Compatibility option; the current workflow does not minimize the receptor with OpenMM.",
    )
    parser.add_argument(
        "--prepare-receptor4",
        type=str,
        default=None,
        help="Explicit path or executable entry point for AutoDockTools prepare_receptor4.py.",
    )
    parser.add_argument(
        "--adt-conda-env",
        type=str,
        default=AUTODOCKTOOLS_CONDA_ENV,
        help="Conda environment containing AutoDockTools/MGLTools.",
    )
    parser.add_argument(
        "--skip-receptor-audit",
        action="store_true",
        help="Skip the read-only missing-atom and pocket-distance audit.",
    )
    parser.add_argument(
        "--prepare-receptor-only",
        action="store_true",
        help="Prepare and audit the receptor without docking the candidate ligands.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root: Path = args.output_root.expanduser().resolve()
    force: bool = bool(args.force)
    segmentation_enabled = SPLIT_RECEPTOR_AT_COORDINATE_GAPS and (
        not bool(args.skip_chain_segmentation)
    )
    receptor_audit_enabled = AUDIT_RECEPTOR_MISSING_ATOMS and (
        not bool(args.skip_receptor_audit)
    )
    print("=" * 90)
    print("NLRP3 AutoDock Vina batch docking")
    print(f"Timestamp     : {now_text()}")
    print("Input source : hardcoded compounds in this Python file (no Excel)")
    print(f"Target       : {TARGET_NAME}")
    print(f"PDB/mmCIF    : {PDB_ID} / {STRUCTURE_FILE_FORMAT}")
    print(f"Output root  : {run_root}")
    print(f"Force rerun  : {force}")
    print("Receptor prep: AutoDockTools prepare_receptor4.py")
    print(f"Gap audit    : {segmentation_enabled}")
    print(f"Missing audit: {receptor_audit_enabled}")
    print(f"ADT env      : {args.adt_conda_env}")
    print("Selected set : ranks 4, 5, 7 (NICERGOLINE, NALFURAFINE, REVUMENIB)")
    print("=" * 90, flush=True)
    check_command_exists("vina")
    check_command_exists("mk_prepare_ligand.py")
    check_command_exists("obabel")
    check_python_package("gemmi", "conda install -c conda-forge gemmi")
    ligands = build_hardcoded_ligands()
    print("[SELECT] The script is configured for these three compounds:")
    for record in ligands:
        print(
            f"[SELECT] rank={record.rank} | {record.chembl_id} | {record.compound_name}"
        )
    if args.min_rank is not None:
        ligands = [x for x in ligands if x.rank is not None and x.rank >= args.min_rank]
    if args.max_rank is not None:
        ligands = [x for x in ligands if x.rank is not None and x.rank <= args.max_rank]
    if args.chembl_id:
        requested = {x.upper() for x in args.chembl_id}
        ligands = [x for x in ligands if x.chembl_id.upper() in requested]
    if not ligands:
        raise RuntimeError(
            "No compounds remain after applying the command-line filters."
        )
    print(f"[INFO] Compounds to process: {len(ligands)}")
    raw_receptor_dir = run_root / "00_raw_receptor"
    prepared_receptor_dir = run_root / "01_prepared_receptor"
    smi_dir = run_root / "02_ligands_smi"
    sdf_dir = run_root / "03_ligands_sdf"
    pdbqt_dir = run_root / "04_ligands_pdbqt"
    docking_dir = run_root / "05_docking_outputs"
    logs_dir = run_root / "06_logs"
    results_dir = run_root / "07_results"
    visual_dir = run_root / "08_visualization_files"
    status_dir = run_root / "09_status"
    for directory in [
        raw_receptor_dir,
        prepared_receptor_dir,
        smi_dir,
        sdf_dir,
        pdbqt_dir,
        docking_dir,
        logs_dir,
        results_dir,
        visual_dir,
        status_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    input_snapshot = results_dir / "input_ligands_snapshot.csv"
    with open(input_snapshot, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "input_row",
                "dta_rank",
                "chembl_id",
                "compound_name",
                "smiles",
                "safe_name",
            ],
        )
        writer.writeheader()
        for ligand in ligands:
            writer.writerow(
                {
                    "input_row": ligand.input_row,
                    "dta_rank": ligand.rank,
                    "chembl_id": ligand.chembl_id,
                    "compound_name": ligand.compound_name,
                    "smiles": ligand.smiles,
                    "safe_name": ligand.safe_name,
                }
            )
    if args.structure_file is not None:
        source_structure = args.structure_file.expanduser().resolve()
        if not source_structure.exists():
            raise FileNotFoundError(
                f"--structure-file does not exist: {source_structure}"
            )
        if source_structure.suffix.lower() not in {".cif", ".mmcif"}:
            raise RuntimeError(
                f"The 9GU4 workflow requires an mmCIF input (.cif or .mmcif). Current file: {source_structure}"
            )
        structure_file = raw_receptor_dir / f"{TARGET_NAME}_{PDB_ID}.cif"
        if force or not structure_file.exists():
            shutil.copy2(source_structure, structure_file)
        print(f"[OK] Using local mmCIF: {source_structure}")
    else:
        structure_file = raw_receptor_dir / f"{TARGET_NAME}_{PDB_ID}.cif"
        structure_url = f"https://files.rcsb.org/download/{PDB_ID}.cif"
        download_file(
            structure_url,
            structure_file,
            overwrite=OVERWRITE_STRUCTURE_DOWNLOAD or force,
        )
    receptor_pdb = prepared_receptor_dir / f"{TARGET_NAME}_{PDB_ID}_protein_only.pdb"
    native_ligand_pdb = (
        prepared_receptor_dir
        / f"{TARGET_NAME}_{PDB_ID}_native_{NATIVE_LIGAND_PDB_ALIAS}.pdb"
    )
    extraction_info = extract_receptor_and_native_ligand_from_mmcif(
        structure_file=structure_file,
        receptor_pdb=receptor_pdb,
        native_ligand_pdb=native_ligand_pdb,
        receptor_chains=RECEPTOR_CHAINS,
        ligand_ccd_id=NATIVE_LIGAND_RESNAME,
        ligand_chain=NATIVE_LIGAND_CHAIN,
        ligand_pdb_alias=NATIVE_LIGAND_PDB_ALIAS,
    )
    segmented_receptor_pdb = (
        prepared_receptor_dir / f"{TARGET_NAME}_{PDB_ID}_protein_segmented.pdb"
    )
    chain_break_report_json = prepared_receptor_dir / "chain_break_report.json"
    if segmentation_enabled:
        segmentation_info = split_receptor_at_coordinate_gaps(
            input_pdb=receptor_pdb,
            output_pdb=segmented_receptor_pdb,
            native_ligand_pdb=native_ligand_pdb,
            report_json=chain_break_report_json,
            force=force,
        )
        print(
            "[TOPOLOGY] The segmented PDB is for inspection only; AutoDockTools uses the original experimental chain."
        )
    else:
        print(
            "[WARN] Coordinate-break inspection was skipped; receptor preparation will continue."
        )
        segmentation_info = {
            "enabled": False,
            "input_pdb": str(receptor_pdb),
            "output_pdb": None,
        }
    missing_atom_audit_json = (
        prepared_receptor_dir / "receptor_missing_atoms_audit.json"
    )
    if receptor_audit_enabled:
        receptor_audit_info = audit_receptor_missing_atoms(
            receptor_pdb=receptor_pdb,
            native_ligand_pdb=native_ligand_pdb,
            report_json=missing_atom_audit_json,
            force=force,
        )
    else:
        receptor_audit_info = {"status": "disabled", "structure_modified": False}
        print("[WARN] The receptor missing-atom and pocket-distance audit was skipped.")
    repair_info = {
        "repair_enabled": False,
        "reason": "Follow published 9GU4 + AutoDockTools workflow; do not reconstruct missing residues/side chains before PDBQT conversion.",
        "input_pdb": str(receptor_pdb),
        "output_pdb": str(receptor_pdb),
    }
    docking_receptor_pdb = receptor_pdb
    auto_center, auto_size, native_span = compute_box_from_native_ligand(
        ligand_pdb=native_ligand_pdb,
        padding=BOX_PADDING,
        min_size=MIN_BOX_SIZE,
        max_size=MAX_BOX_SIZE,
    )
    center = BOX_CENTER_OVERRIDE if BOX_CENTER_OVERRIDE is not None else auto_center
    box_size = BOX_SIZE_OVERRIDE if BOX_SIZE_OVERRIDE is not None else auto_size
    config_file = prepared_receptor_dir / "vina_config.txt"
    write_vina_config(config_file, center=center, box_size=box_size)
    autodocktools_receptor_dir = prepared_receptor_dir / "autodocktools_receptor"
    receptor_pdbqt, receptor_prep_info = prepare_receptor(
        receptor_pdb=docking_receptor_pdb,
        receptor_dir=autodocktools_receptor_dir,
        target_name=f"{TARGET_NAME}_{PDB_ID}_ADT",
        center=center,
        box_size=box_size,
        force=force,
        explicit_prepare_receptor4=args.prepare_receptor4,
        autodocktools_conda_env=args.adt_conda_env,
    )
    run_metadata = {
        "created_at": now_text(),
        "script": str(Path(__file__).resolve()),
        "input_file": None,
        "input_source": "hardcoded_compounds_in_python",
        "hardcoded_compounds": list(HARDCODED_COMPOUNDS),
        "input_count": len(ligands),
        "target_name": TARGET_NAME,
        "pdb_id": PDB_ID,
        "structure_format": STRUCTURE_FILE_FORMAT,
        "structure_file": str(structure_file),
        "extracted_receptor_pdb": str(receptor_pdb),
        "segmented_receptor_pdb": (
            str(segmented_receptor_pdb) if segmentation_enabled else None
        ),
        "chain_break_report": (
            str(chain_break_report_json) if segmentation_enabled else None
        ),
        "segmentation_info": segmentation_info,
        "docking_receptor_pdb": str(docking_receptor_pdb),
        "receptor_missing_atoms_audit": (
            str(missing_atom_audit_json) if receptor_audit_enabled else None
        ),
        "receptor_audit_info": receptor_audit_info,
        "repair_info": repair_info,
        "receptor_preparation_backend": "AutoDockTools prepare_receptor4.py",
        "receptor_preparation_info": receptor_prep_info,
        "receptor_pdbqt": str(receptor_pdbqt),
        "receptor_chains": RECEPTOR_CHAINS,
        "native_ligand_ccd_id": NATIVE_LIGAND_RESNAME,
        "native_ligand_display_name": NATIVE_LIGAND_DISPLAY_NAME,
        "native_ligand_pdb_alias": NATIVE_LIGAND_PDB_ALIAS,
        "native_ligand_pdb": str(native_ligand_pdb),
        "native_ligand_span": native_span,
        "box_center": center,
        "box_size": box_size,
        "box_padding": BOX_PADDING,
        "exhaustiveness": EXHAUSTIVENESS,
        "num_modes": NUM_MODES,
        "energy_range": ENERGY_RANGE,
        "cpu": CPU,
        "seed": SEED,
        "ligand_pH": LIGAND_PH,
        "repair_receptor_with_pdbfixer": False,
        "receptor_coordinates_modified": False,
        "autodocktools_repairs": AUTODOCKTOOLS_REPAIRS,
        "autodocktools_cleanup": AUTODOCKTOOLS_CLEANUP,
        "autodocktools_conda_environment": args.adt_conda_env,
        "meeko_used_for_receptor": False,
        "meeko_used_for_ligands": True,
        "create_visualization_files": CREATE_VISUALIZATION_FILES,
        "extraction_info": extraction_info,
        "force": force,
    }
    atomic_write_json(run_metadata, results_dir / "run_metadata.json")
    atomic_write_json(
        {
            "center_x": center[0],
            "center_y": center[1],
            "center_z": center[2],
            "size_x": box_size[0],
            "size_y": box_size[1],
            "size_z": box_size[2],
            "derived_from": f"PDB {PDB_ID}, ligand {NATIVE_LIGAND_DISPLAY_NAME} (CCD {NATIVE_LIGAND_RESNAME})",
            "native_ligand_span": native_span,
        },
        results_dir / "box_info.json",
    )
    if args.prepare_receptor_only:
        print("\n" + "=" * 90)
        print("[DONE] Receptor audit and AutoDockTools PDBQT preparation completed.")
        print(f"[RESULT] Extracted receptor: {receptor_pdb}")
        if segmentation_enabled:
            print(f"[RESULT] Segmented inspection copy: {segmented_receptor_pdb}")
            print(f"[RESULT] Chain-break report: {chain_break_report_json}")
        print(f"[RESULT] Experimental receptor supplied to ADT: {docking_receptor_pdb}")
        print(f"[RESULT] AutoDockTools receptor PDBQT: {receptor_pdbqt}")
        if receptor_audit_enabled:
            print(f"[RESULT] Receptor missing-atom audit: {missing_atom_audit_json}")
        print(
            f"[RESULT] AutoDockTools preparation log: {autodocktools_receptor_dir / 'prepare_receptor4.log'}"
        )
        print(f"[RESULT] Docking box: {results_dir / 'box_info.json'}")
        print("=" * 90)
        return
    result_rows: List[Dict[str, object]] = []
    for index, ligand in enumerate(ligands, start=1):
        print("\n" + "-" * 90)
        print(
            f"[LIGAND {index}/{len(ligands)}] rank={ligand.rank} | {ligand.chembl_id} | {ligand.compound_name}"
        )
        print(f"[SMILES] {ligand.smiles}", flush=True)
        smi_file = smi_dir / f"{ligand.safe_name}.smi"
        sdf_file = sdf_dir / f"{ligand.safe_name}.sdf"
        ligand_pdbqt = pdbqt_dir / f"{ligand.safe_name}.pdbqt"
        vina_out = docking_dir / f"{ligand.safe_name}_vina_all_poses.pdbqt"
        vina_log = logs_dir / f"{ligand.safe_name}_vina.log"
        gen3d_log = logs_dir / f"{ligand.safe_name}_obabel_gen3d.log"
        meeko_log = logs_dir / f"{ligand.safe_name}_meeko_ligand.log"
        best_pose_pdbqt = visual_dir / f"{ligand.safe_name}_best_pose.pdbqt"
        best_pose_sdf = visual_dir / f"{ligand.safe_name}_best_pose.sdf"
        best_pose_pdb = visual_dir / f"{ligand.safe_name}_best_pose.pdb"
        complex_pdb = visual_dir / f"{ligand.safe_name}_complex.pdb"
        pymol_script = visual_dir / f"{ligand.safe_name}.pml"
        conversion_log = logs_dir / f"{ligand.safe_name}_conversion.log"
        status_json = status_dir / f"{ligand.safe_name}.json"
        existing_status = load_existing_status(status_json)
        if (
            RESUME
            and (not force)
            and existing_status
            and (existing_status.get("status") == "success")
            and vina_out.exists()
        ):
            print(f"[RESUME] Skipping completed result: {ligand.safe_name}")
            result_rows.append(existing_status)
            update_summary_files(result_rows, results_dir)
            continue
        started_at = now_text()
        row: Dict[str, object] = {
            "input_row": ligand.input_row,
            "dta_rank": ligand.rank,
            "chembl_id": ligand.chembl_id,
            "compound_name": ligand.compound_name,
            "file_safe_name": ligand.safe_name,
            "smiles": ligand.smiles,
            "vina_score_kcal_mol": None,
            "status": "running",
            "started_at": started_at,
            "finished_at": "",
            "error": "",
            "warning": "",
            "ligand_smi": str(smi_file),
            "ligand_sdf": str(sdf_file),
            "ligand_pdbqt": str(ligand_pdbqt),
            "vina_all_poses_pdbqt": str(vina_out),
            "vina_log": str(vina_log),
            "best_pose_pdbqt": str(best_pose_pdbqt),
            "best_pose_sdf": str(best_pose_sdf),
            "best_pose_pdb": str(best_pose_pdb),
            "complex_pdb": str(complex_pdb),
            "pymol_script": str(pymol_script),
            "status_json": str(status_json),
        }
        atomic_write_json(row, status_json)
        warnings: List[str] = []
        try:
            generate_ligand_3d_sdf(
                record=ligand,
                smi_file=smi_file,
                sdf_file=sdf_file,
                log_file=gen3d_log,
                force=force,
            )
            prepare_ligand_pdbqt(
                sdf_file=sdf_file,
                pdbqt_file=ligand_pdbqt,
                log_file=meeko_log,
                force=force,
            )
            best_score = run_vina_docking(
                receptor_pdbqt=receptor_pdbqt,
                ligand_pdbqt=ligand_pdbqt,
                config_file=config_file,
                out_pdbqt=vina_out,
                log_file=vina_log,
                force=force,
            )
            row["vina_score_kcal_mol"] = best_score
            if CREATE_VISUALIZATION_FILES:
                try:
                    convert_best_pose(
                        all_poses_pdbqt=vina_out,
                        best_pose_pdbqt=best_pose_pdbqt,
                        best_pose_sdf=best_pose_sdf,
                        best_pose_pdb=best_pose_pdb,
                        conversion_log=conversion_log,
                    )
                    combine_receptor_and_ligand_pdb(
                        receptor_pdb=docking_receptor_pdb,
                        ligand_pdb=best_pose_pdb,
                        complex_pdb=complex_pdb,
                    )
                    write_pymol_script(
                        receptor_pdb=docking_receptor_pdb,
                        ligand_pdb=best_pose_pdb,
                        output_pml=pymol_script,
                        compound_name=ligand.compound_name,
                    )
                except Exception as visualization_error:
                    warning = f"Visualization generation failed, but the Vina result is valid: {visualization_error}"
                    warnings.append(warning)
                    print(f"[WARN] {warning}")
            row["status"] = "success"
            print(f"[SUCCESS] {ligand.compound_name}: {best_score:.3f} kcal/mol")
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
            print(f"[ERROR] {ligand.compound_name}：{exc}", flush=True)
        row["warning"] = " | ".join(warnings)
        row["finished_at"] = now_text()
        atomic_write_json(row, status_json)
        result_rows.append(row)
        update_summary_files(result_rows, results_dir)
    all_csv, sorted_csv = update_summary_files(result_rows, results_dir)
    success_rows = [row for row in result_rows if row.get("status") == "success"]
    failed_rows = [row for row in result_rows if row.get("status") == "failed"]
    summary = {
        "finished_at": now_text(),
        "input_count": len(ligands),
        "success_count": len(success_rows),
        "failed_count": len(failed_rows),
        "all_results_csv": str(all_csv),
        "sorted_results_csv": str(sorted_csv),
        "failed_compounds": [
            {
                "rank": row.get("dta_rank"),
                "chembl_id": row.get("chembl_id"),
                "compound_name": row.get("compound_name"),
                "error": row.get("error"),
            }
            for row in failed_rows
        ],
    }
    atomic_write_json(summary, results_dir / "run_summary.json")
    print("\n" + "=" * 90)
    print("[DONE] NLRP3 batch docking finished")
    print(
        f"[COUNT] total={len(ligands)}, success={len(success_rows)}, failed={len(failed_rows)}"
    )
    print(f"[RESULT] All results: {all_csv}")
    print(f"[RESULT] Score ranking: {sorted_csv}")
    print(f"[RESULT] Run summary: {results_dir / 'run_summary.json'}")
    print(f"[RESULT] Docking box: {results_dir / 'box_info.json'}")
    print("=" * 90)
    ranked_success = sorted(
        [row for row in success_rows if row.get("vina_score_kcal_mol") is not None],
        key=lambda row: float(row["vina_score_kcal_mol"]),
    )
    print("\nTop Vina scores:")
    for row in ranked_success[:20]:
        print(
            f"rank={row.get('dta_rank')}\t{row.get('chembl_id')}\t{row.get('compound_name')}\t{row.get('vina_score_kcal_mol')} kcal/mol"
        )
    if failed_rows:
        print("\nFailed compounds:")
        for row in failed_rows:
            print(
                f"rank={row.get('dta_rank')}\t{row.get('chembl_id')}\t{row.get('compound_name')}\t{row.get('error')}"
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\n[INTERRUPTED] Interrupted by the user. Completed states and summary files were saved.",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except Exception as error:
        print(f"\n[FATAL] {error}", file=sys.stderr)
        raise
