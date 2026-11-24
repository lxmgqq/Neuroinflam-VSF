#!/usr/bin/env python3
"""Redock the native NP3-253 ligand into NLRP3 structure 9GU4 and calculate pose RMSD."""
from __future__ import annotations
import argparse
import csv
import io
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
PDB_ID = "9GU4"
LIGAND_CCD_ID = "A1IPJ"
LIGAND_NAME = "NP3-253"
LIGAND_LABEL_ASYM_ID = "B"
LIGAND_AUTH_SEQ_ID = 701
# Box coordinates match the native-ligand pocket used for candidate docking.
DEFAULT_CENTER = (16.207, 34.721, -6.404)
DEFAULT_BOX_SIZE = (25.436, 26.033, 24.717)
DEFAULT_EXHAUSTIVENESS = 32
DEFAULT_NUM_MODES = 10
DEFAULT_ENERGY_RANGE = 4.0
DEFAULT_CPU = 8
DEFAULT_SEED = 2026
DEFAULT_PH = 7.4
DEFAULT_RMSD_CUTOFF = 2.0
# Generated artifacts stay under the repository data directory and are ignored by Git.
DEFAULT_DOCKING_ROOT = REPO_ROOT / "data" / "docking_results"
DEFAULT_CANDIDATE_OUTPUT_DIR = DEFAULT_DOCKING_ROOT / "NLRP3_candidates"
DEFAULT_OUTPUT_DIR = DEFAULT_DOCKING_ROOT / "redocking" / "9GU4_NP3_253"


def print_header(title: str) -> None:
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)


def print_subheader(title: str) -> None:
    print("\n" + "-" * 92)
    print(title)
    print("-" * 92)


def fail(message: str, exit_code: int = 1) -> "None":
    print(f"\n[FATAL] {message}", file=sys.stderr, flush=True)
    raise SystemExit(exit_code)


def require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        fail(f"Missing {description}：{path}")
    if path.stat().st_size == 0:
        fail(f"{description} is empty: {path}")
    return path


def find_default_paths() -> Tuple[Path, Path]:
    receptor = (
        DEFAULT_CANDIDATE_OUTPUT_DIR
        / "01_prepared_receptor"
        / "autodocktools_receptor"
        / "NLRP3_9GU4_ADT.pdbqt"
    )
    config = DEFAULT_CANDIDATE_OUTPUT_DIR / "01_prepared_receptor" / "vina_config.txt"
    return (receptor, config)


def parse_vina_config(path: Optional[Path]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    if path is None or not path.is_file():
        return values
    pattern = re.compile(
        "^\\s*([A-Za-z_][A-Za-z0-9_]*)\\s*(?:=|\\s)\\s*([-+]?\\d+(?:\\.\\d+)?(?:[eE][-+]?\\d+)?)"
    )
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = pattern.match(line)
        if match:
            try:
                values[match.group(1).lower()] = float(match.group(2))
            except ValueError:
                pass
    return values


def resolve_parameter(
    cli_value: Optional[float], config: Dict[str, float], key: str, default: float
) -> float:
    if cli_value is not None:
        return cli_value
    if key.lower() in config:
        return config[key.lower()]
    return default


def download_text(url: str, timeout: int = 45) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "NLRP3-NP3-253-redocking/1.0",
            "Accept": "chemical/x-mdl-sdfile,text/plain,*/*",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
        charset = response.headers.get_content_charset() or "utf-8"
    return raw.decode(charset, errors="replace")


def fetch_native_ligand_sdf(timeout: int = 45) -> Tuple[str, str]:
    base = f"https://models.rcsb.org/v1/{PDB_ID.lower()}/ligand"
    queries = [
        {
            "label_comp_id": LIGAND_CCD_ID,
            "encoding": "sdf",
            "copy_all_categories": "false",
            "download": "true",
        },
        {
            "label_asym_id": LIGAND_LABEL_ASYM_ID,
            "auth_seq_id": str(LIGAND_AUTH_SEQ_ID),
            "encoding": "sdf",
            "copy_all_categories": "false",
            "download": "true",
        },
    ]
    errors: List[str] = []
    for query in queries:
        url = base + "?" + urllib.parse.urlencode(query)
        try:
            text = download_text(url, timeout=timeout)
            if "$$$$" not in text or "M  END" not in text:
                raise RuntimeError("The response does not appear to be a valid SDF")
            return (text, url)
        except Exception as exc:
            errors.append(f"{url} -> {type(exc).__name__}: {exc}")
    fail(
        "Could not retrieve the NP3-253 instance coordinates from 9GU4 through the RCSB ModelServer.\n"
        + "\n".join(errors)
    )


def load_first_valid_sdf_molecule(sdf_text: str):
    try:
        from rdkit import Chem
    except ImportError:
        fail("RDKit is required. Install it in the autodock_vina environment.")
    supplier = Chem.ForwardSDMolSupplier(
        io.BytesIO(sdf_text.encode("utf-8")),
        sanitize=True,
        removeHs=False,
        strictParsing=True,
    )
    valid = [mol for mol in supplier if mol is not None]
    if not valid:
        fail("RDKit could not parse the SDF returned by RCSB.")
    for mol in valid:
        heavy_count = sum((1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1))
        if heavy_count == 27:
            mol.SetProp("_Name", LIGAND_NAME)
            return mol
    mol = valid[0]
    mol.SetProp("_Name", LIGAND_NAME)
    return mol


def protonate_with_openbabel_in_memory(mol, ph: float):
    from rdkit import Chem

    obabel = shutil.which("obabel")
    if not obabel:
        print(
            "[WARN] obabel was not found; using RDKit AddHs without pH-dependent protonation.",
            flush=True,
        )
        return Chem.AddHs(Chem.Mol(mol), addCoords=True)
    mol_block = Chem.MolToMolBlock(mol)
    cmd = [obabel, "-isdf", "-osdf", "-p", str(ph), "-h"]
    process = subprocess.run(
        cmd,
        input=mol_block,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0 or "M  END" not in process.stdout:
        print(
            f"[WARN] Open Babel protonation failed; falling back to RDKit AddHs.\n[WARN] Command: {' '.join(cmd)}\n[WARN] stderr：{process.stderr[-1500:]}",
            flush=True,
        )
        return Chem.AddHs(Chem.Mol(mol), addCoords=True)
    protonated = Chem.MolFromMolBlock(
        process.stdout, sanitize=True, removeHs=False, strictParsing=True
    )
    if protonated is None:
        print(
            "[WARN] RDKit could not parse the Open Babel output; falling back to RDKit AddHs.",
            flush=True,
        )
        return Chem.AddHs(Chem.Mol(mol), addCoords=True)
    protonated.SetProp("_Name", LIGAND_NAME)
    return protonated


def prepare_ligand_pdbqt_in_memory(mol) -> str:
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
    except ImportError:
        return prepare_ligand_pdbqt_with_temp_cli(mol)
    try:
        preparator = MoleculePreparation()
        if hasattr(preparator, "prepare"):
            setups = preparator.prepare(mol)
        else:
            setups = preparator(mol)
        if not setups:
            raise RuntimeError("Meeko did not generate a MoleculeSetup object")
        written = PDBQTWriterLegacy.write_string(setups[0])
        if isinstance(written, tuple):
            pdbqt_string, is_ok, error_msg = written
        else:
            pdbqt_string, is_ok, error_msg = (written, bool(written), "")
        if not is_ok or not str(pdbqt_string).strip():
            raise RuntimeError(error_msg or "Meeko PDBQTWriterLegacy reported failure")
        return str(pdbqt_string)
    except Exception as exc:
        print(
            f"[WARN] Meeko Python API preparation failed; trying mk_prepare_ligand.py: {exc}",
            flush=True,
        )
        return prepare_ligand_pdbqt_with_temp_cli(mol)


def prepare_ligand_pdbqt_with_temp_cli(mol) -> str:
    from rdkit import Chem

    executable = shutil.which("mk_prepare_ligand.py")
    if not executable:
        fail(
            "The Meeko Python API and mk_prepare_ligand.py are unavailable.\nActivate the autodock_vina environment."
        )
    with tempfile.TemporaryDirectory(prefix="np3_redock_ligand_") as tmp:
        tmpdir = Path(tmp)
        sdf_path = tmpdir / "NP3-253_crystal.sdf"
        pdbqt_path = tmpdir / "NP3-253.pdbqt"
        writer = Chem.SDWriter(str(sdf_path))
        writer.write(mol)
        writer.close()
        process = subprocess.run(
            [executable, "-i", str(sdf_path), "-o", str(pdbqt_path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if process.returncode != 0 or not pdbqt_path.is_file():
            fail(
                f"mk_prepare_ligand.py failed to prepare NP3-253.\nOutput:\n{process.stdout[-4000:]}"
            )
        return pdbqt_path.read_text(encoding="utf-8", errors="ignore")


def parse_vina_affinities_from_pdbqt(pdbqt_text: str) -> List[float]:
    pattern = re.compile(
        "REMARK\\s+VINA\\s+RESULT:\\s*([-+]?\\d+(?:\\.\\d+)?)", flags=re.IGNORECASE
    )
    return [float(x) for x in pattern.findall(pdbqt_text)]


def run_vina_python_api(
    receptor: Path,
    ligand_pdbqt: str,
    center: Sequence[float],
    box_size: Sequence[float],
    exhaustiveness: int,
    num_modes: int,
    energy_range: float,
    cpu: int,
    seed: int,
) -> Tuple[str, List[float], str]:
    from vina import Vina

    v = Vina(sf_name="vina", cpu=cpu, seed=seed, verbosity=1)
    v.set_receptor(str(receptor))
    v.set_ligand_from_string(ligand_pdbqt)
    v.compute_vina_maps(center=list(center), box_size=list(box_size))
    v.dock(exhaustiveness=exhaustiveness, n_poses=num_modes)
    poses = v.poses(n_poses=num_modes, energy_range=energy_range)
    if isinstance(poses, bytes):
        poses = poses.decode("utf-8", errors="replace")
    affinities: List[float] = []
    try:
        energies = v.energies(n_poses=num_modes, energy_range=energy_range)
        for row in energies:
            affinities.append(float(row[0]))
    except Exception:
        affinities = parse_vina_affinities_from_pdbqt(poses)
    return (poses, affinities, "Vina Python API (in memory)")


def run_vina_cli_temp(
    receptor: Path,
    ligand_pdbqt: str,
    center: Sequence[float],
    box_size: Sequence[float],
    exhaustiveness: int,
    num_modes: int,
    energy_range: float,
    cpu: int,
    seed: int,
) -> Tuple[str, List[float], str]:
    vina_exe = shutil.which("vina")
    if not vina_exe:
        fail("The Vina Python API and vina command are both unavailable.")
    with tempfile.TemporaryDirectory(prefix="np3_redock_vina_") as tmp:
        tmpdir = Path(tmp)
        ligand_path = tmpdir / "NP3-253.pdbqt"
        output_path = tmpdir / "NP3-253_redocked.pdbqt"
        ligand_path.write_text(ligand_pdbqt, encoding="utf-8")
        cmd = [
            vina_exe,
            "--receptor",
            str(receptor),
            "--ligand",
            str(ligand_path),
            "--center_x",
            str(center[0]),
            "--center_y",
            str(center[1]),
            "--center_z",
            str(center[2]),
            "--size_x",
            str(box_size[0]),
            "--size_y",
            str(box_size[1]),
            "--size_z",
            str(box_size[2]),
            "--exhaustiveness",
            str(exhaustiveness),
            "--num_modes",
            str(num_modes),
            "--energy_range",
            str(energy_range),
            "--cpu",
            str(cpu),
            "--seed",
            str(seed),
            "--out",
            str(output_path),
        ]
        print("\n[CMD] " + " ".join(cmd), flush=True)
        process = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        print(process.stdout, end="" if process.stdout.endswith("\n") else "\n")
        if process.returncode != 0 or not output_path.is_file():
            fail(
                f"Vina redocking failed.\nReturn code={process.returncode}\nLog tail:\n{process.stdout[-5000:]}"
            )
        poses = output_path.read_text(encoding="utf-8", errors="ignore")
        affinities = parse_vina_affinities_from_pdbqt(poses)
        return (
            poses,
            affinities,
            "Vina command line (temporary files cleaned automatically)",
        )


def run_vina(
    receptor: Path,
    ligand_pdbqt: str,
    center: Sequence[float],
    box_size: Sequence[float],
    exhaustiveness: int,
    num_modes: int,
    energy_range: float,
    cpu: int,
    seed: int,
) -> Tuple[str, List[float], str]:
    try:
        __import__("vina")
    except ImportError:
        print("[INFO] Vina Python API is unavailable; using the command-line backend.")
        return run_vina_cli_temp(
            receptor,
            ligand_pdbqt,
            center,
            box_size,
            exhaustiveness,
            num_modes,
            energy_range,
            cpu,
            seed,
        )
    try:
        return run_vina_python_api(
            receptor,
            ligand_pdbqt,
            center,
            box_size,
            exhaustiveness,
            num_modes,
            energy_range,
            cpu,
            seed,
        )
    except Exception as exc:
        print(
            f"[WARN] Vina Python API failed; switching to the command-line backend: {exc}",
            flush=True,
        )
        return run_vina_cli_temp(
            receptor,
            ligand_pdbqt,
            center,
            box_size,
            exhaustiveness,
            num_modes,
            energy_range,
            cpu,
            seed,
        )


def docked_pdbqt_to_rdkit_mol(pdbqt_text: str):
    try:
        from meeko import PDBQTMolecule, RDKitMolCreate
    except ImportError:
        fail("Meeko is required to calculate RMSD but could not be imported.")
    errors: List[str] = []
    for is_dlg in (False, True):
        try:
            pdbqt_mol = PDBQTMolecule(pdbqt_text, is_dlg=is_dlg, skip_typing=True)
            try:
                mols = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol, keep_flexres=False)
            except TypeError:
                mols = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
            valid = [mol for mol in mols if mol is not None]
            if valid:
                return valid[0]
            errors.append(f"is_dlg={is_dlg}: RDKitMolCreate returned no molecule")
        except Exception as exc:
            errors.append(f"is_dlg={is_dlg}: {type(exc).__name__}: {exc}")
    fail(
        "Meeko could not reconstruct an RDKit molecule with correct bond orders from the Vina output.\n"
        + "\n".join(errors)
    )


def prepare_heavy_molecule(mol):
    from rdkit import Chem

    heavy = Chem.RemoveHs(Chem.Mol(mol), sanitize=True)
    if heavy.GetNumConformers() == 0:
        fail("The molecule has no 3D coordinates; RMSD cannot be calculated.")
    return heavy


def enumerate_full_graph_mappings(
    ref_heavy, dock_heavy
) -> Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]], int]:
    from rdkit import Chem
    from rdkit.Chem import rdFMCS

    if ref_heavy.GetNumAtoms() != dock_heavy.GetNumAtoms():
        fail(
            f"The crystal and redocked ligands have different heavy-atom counts: reference={ref_heavy.GetNumAtoms()}, docked={dock_heavy.GetNumAtoms()}。"
        )
    mcs = rdFMCS.FindMCS(
        [ref_heavy, dock_heavy],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        matchValences=False,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        timeout=20,
    )
    if mcs.canceled:
        fail("The MCS search for RMSD atom mapping timed out.")
    if mcs.numAtoms != ref_heavy.GetNumAtoms():
        fail(
            f"A complete heavy-atom mapping could not be established between the crystal and redocked ligands: MCS={mcs.numAtoms}, total={ref_heavy.GetNumAtoms()}。"
        )
    query = Chem.MolFromSmarts(mcs.smartsString)
    if query is None:
        fail("Could not create an atom-mapping query from the MCS SMARTS.")
    max_matches = 20000
    ref_matches = list(
        ref_heavy.GetSubstructMatches(
            query, uniquify=False, useChirality=False, maxMatches=max_matches
        )
    )
    dock_matches = list(
        dock_heavy.GetSubstructMatches(
            query, uniquify=False, useChirality=False, maxMatches=max_matches
        )
    )
    if not ref_matches or not dock_matches:
        fail("A complete MCS was found, but atom mappings could not be enumerated.")
    return (ref_matches, dock_matches, mcs.numAtoms)


def symmetry_corrected_fixed_frame_rmsds(
    ref_mol, docked_mol
) -> Tuple[List[float], int]:
    ref_heavy = prepare_heavy_molecule(ref_mol)
    dock_heavy = prepare_heavy_molecule(docked_mol)
    ref_matches, dock_matches, n_atoms = enumerate_full_graph_mappings(
        ref_heavy, dock_heavy
    )
    ref_conf = ref_heavy.GetConformer(0)
    rmsds: List[float] = []
    for conf_id in range(dock_heavy.GetNumConformers()):
        dock_conf = dock_heavy.GetConformer(conf_id)
        best_sq = math.inf
        for ref_match in ref_matches:
            for dock_match in dock_matches:
                sq = 0.0
                for q_idx in range(n_atoms):
                    rp = ref_conf.GetAtomPosition(ref_match[q_idx])
                    dp = dock_conf.GetAtomPosition(dock_match[q_idx])
                    dx = rp.x - dp.x
                    dy = rp.y - dp.y
                    dz = rp.z - dp.z
                    sq += dx * dx + dy * dy + dz * dz
                if sq < best_sq:
                    best_sq = sq
        rmsds.append(math.sqrt(best_sq / n_atoms))
    return (rmsds, n_atoms)


def format_float(value: Optional[float], digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def prepare_output_directories(output_dir: Path) -> Dict[str, Path]:
    root = output_dir.expanduser().resolve()
    directories = {
        "root": root,
        "inputs": root / "00_inputs",
        "docking": root / "01_docking",
        "results": root / "02_results",
        "visualization": root / "03_visualization",
        "logs": root / "04_logs",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def write_rdkit_sdf(mol, path: Path) -> None:
    from rdkit import Chem

    writer = Chem.SDWriter(str(path))
    if writer is None:
        fail(f"Could not create an SDF writer: {path}")
    writer.write(mol)
    writer.close()
    require_file(path, "SDF output")


def molecule_with_single_conformer(mol, conf_id: int, name: str):
    from rdkit import Chem

    if conf_id < 0 or conf_id >= mol.GetNumConformers():
        fail(
            f"Conformer index is out of range: conf_id={conf_id}, n_conformers={mol.GetNumConformers()}"
        )
    single = Chem.Mol(mol)
    conformer = Chem.Conformer(mol.GetConformer(conf_id))
    single.RemoveAllConformers()
    single.AddConformer(conformer, assignId=True)
    single.SetProp("_Name", name)
    return single


def save_docked_pose_files(
    docked_mol,
    affinities: Sequence[float],
    rmsds: Sequence[float],
    directories: Dict[str, Path],
    top1_pose: int,
    lowest_rmsd_pose: int,
) -> Dict[str, Path]:
    from rdkit import Chem

    all_sdf = directories["docking"] / "NP3_253_redocked_all_poses.sdf"
    writer = Chem.SDWriter(str(all_sdf))
    if writer is None:
        fail(f"Could not create an SDF writer: {all_sdf}")
    n_conformers = docked_mol.GetNumConformers()
    for conf_id in range(n_conformers):
        pose_number = conf_id + 1
        pose = molecule_with_single_conformer(
            docked_mol, conf_id, f"NP3-253_redocked_pose_{pose_number}"
        )
        pose.SetIntProp("pose", pose_number)
        if conf_id < len(affinities):
            pose.SetDoubleProp("vina_affinity_kcal_mol", float(affinities[conf_id]))
        if conf_id < len(rmsds):
            pose.SetDoubleProp("fixed_frame_heavy_atom_rmsd_A", float(rmsds[conf_id]))
        writer.write(pose)
    writer.close()
    require_file(all_sdf, "all redocked poses SDF")

    def save_one(pose_number: int, stem: str) -> Tuple[Path, Path]:
        conf_id = pose_number - 1
        pose = molecule_with_single_conformer(docked_mol, conf_id, f"NP3-253_{stem}")
        pose.SetIntProp("pose", pose_number)
        if conf_id < len(affinities):
            pose.SetDoubleProp("vina_affinity_kcal_mol", float(affinities[conf_id]))
        if conf_id < len(rmsds):
            pose.SetDoubleProp("fixed_frame_heavy_atom_rmsd_A", float(rmsds[conf_id]))
        sdf_path = directories["docking"] / f"NP3_253_{stem}.sdf"
        pdb_path = directories["visualization"] / f"NP3_253_{stem}.pdb"
        write_rdkit_sdf(pose, sdf_path)
        Chem.MolToPDBFile(pose, str(pdb_path))
        require_file(pdb_path, "redocked pose PDB")
        return (sdf_path, pdb_path)

    top1_sdf, top1_pdb = save_one(top1_pose, "redocked_top1")
    lowest_sdf, lowest_pdb = save_one(
        lowest_rmsd_pose, f"redocked_lowest_rmsd_pose{lowest_rmsd_pose:02d}"
    )
    return {
        "all_poses_sdf": all_sdf,
        "top1_sdf": top1_sdf,
        "top1_pdb": top1_pdb,
        "lowest_rmsd_sdf": lowest_sdf,
        "lowest_rmsd_pdb": lowest_pdb,
    }


def find_and_copy_protein_for_visualization(
    receptor: Path, visualization_dir: Path
) -> Path:
    prepared_dir = receptor.parent.parent
    candidates = [
        prepared_dir / "NLRP3_9GU4_protein_only.pdb",
        prepared_dir / "NLRP3_9GU4_protein_segmented.pdb",
    ]
    source = next((path for path in candidates if path.is_file()), receptor)
    suffix = ".pdb" if source.suffix.lower() == ".pdb" else ".pdbqt"
    destination = visualization_dir / f"NLRP3_9GU4_receptor{suffix}"
    shutil.copy2(source, destination)
    return require_file(destination, "PyMOL receptor file")


def create_pymol_overlay_script(
    protein_path: Path,
    crystal_pdb: Path,
    top1_pdb: Path,
    lowest_rmsd_pdb: Path,
    pml_path: Path,
    top1_rmsd: float,
    lowest_rmsd: float,
    lowest_rmsd_pose: int,
) -> Dict[str, Path]:

    def pp(path: Path) -> str:
        return str(path.resolve()).replace("\\", "/")

    visualization_dir = pml_path.parent
    top1_png = visualization_dir / "NP3_253_overlay_top1.png"
    lowest_png = (
        visualization_dir
        / f"NP3_253_overlay_lowest_rmsd_pose{lowest_rmsd_pose:02d}.png"
    )
    pse_path = visualization_dir / "NP3_253_redocking_overlay.pse"
    script = f'# Auto-generated PyMOL overlay for NLRP3 9GU4 / NP3-253 redocking\n# Crystal pose: hot pink carbon atoms\n# Vina redocked pose: cyan carbon atoms\n# No fit/align command is used; both poses remain in the fixed receptor frame.\n# Top-1 heavy-atom RMSD: {top1_rmsd:.3f} A\n# Lowest RMSD: pose {lowest_rmsd_pose}, {lowest_rmsd:.3f} A\n\nreinitialize\nload {pp(protein_path)}, receptor\nload {pp(crystal_pdb)}, crystal_NP3_253\nload {pp(top1_pdb)}, redocked_top1\nload {pp(lowest_rmsd_pdb)}, redocked_lowest_rmsd\n\nhide everything\nbg_color white\nset ray_opaque_background, on\nset orthoscopic, on\nset antialias, 2\nset ray_shadows, off\nset depth_cue, off\nset cartoon_transparency, 0.18\nset stick_radius, 0.20\nset scene_buttons, on\n\nshow cartoon, receptor\ncolor gray80, receptor\n\nselect pocket, byres (receptor within 4.0 of (crystal_NP3_253 or redocked_top1))\nshow sticks, pocket\ncolor marine, pocket and elem C\ncolor red, pocket and elem O\ncolor blue, pocket and elem N\ncolor yellow, pocket and elem S\n\nshow sticks, crystal_NP3_253\ncolor hotpink, crystal_NP3_253 and elem C\ncolor red, crystal_NP3_253 and elem O\ncolor blue, crystal_NP3_253 and elem N\ncolor green, crystal_NP3_253 and (elem F or elem Cl or elem Br or elem I)\n\nshow sticks, redocked_top1\ncolor cyan, redocked_top1 and elem C\ncolor red, redocked_top1 and elem O\ncolor blue, redocked_top1 and elem N\ncolor green, redocked_top1 and (elem F or elem Cl or elem Br or elem I)\n\ndisable redocked_lowest_rmsd\norient crystal_NP3_253 or redocked_top1\nzoom crystal_NP3_253 or redocked_top1, 8\nscene 01_top1_overlay, store\npng {pp(top1_png)}, 2400, 1800, 300, 1\n\ndisable redocked_top1\nenable redocked_lowest_rmsd\nshow sticks, redocked_lowest_rmsd\ncolor cyan, redocked_lowest_rmsd and elem C\ncolor red, redocked_lowest_rmsd and elem O\ncolor blue, redocked_lowest_rmsd and elem N\ncolor green, redocked_lowest_rmsd and (elem F or elem Cl or elem Br or elem I)\norient crystal_NP3_253 or redocked_lowest_rmsd\nzoom crystal_NP3_253 or redocked_lowest_rmsd, 8\nscene 02_lowest_rmsd_overlay, store\npng {pp(lowest_png)}, 2400, 1800, 300, 1\n\nscene 01_top1_overlay, recall\nsave {pp(pse_path)}\nprint("NP3-253 redocking overlay finished.")\n'
    pml_path.write_text(script, encoding="utf-8")
    require_file(pml_path, "PyMOL overlay script")
    return {
        "pml": pml_path,
        "top1_png": top1_png,
        "lowest_rmsd_png": lowest_png,
        "pse": pse_path,
    }


def render_pymol_script(pml_path: Path, log_path: Path) -> bool:
    pymol_exe = shutil.which("pymol")
    if not pymol_exe:
        print(
            "[WARN] --render-pymol was requested, but pymol was not found. PML and PDB files were retained for manual rendering."
        )
        return False
    process = subprocess.run(
        [pymol_exe, "-cq", str(pml_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(process.stdout or "", encoding="utf-8")
    if process.returncode != 0:
        print(
            f"[WARN] PyMOL rendering failed with return code {process.returncode}; see: {log_path}"
        )
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    default_receptor, default_config = find_default_paths()
    parser = argparse.ArgumentParser(
        description="Redock the 9GU4 co-crystal ligand NP3-253 and save all outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--receptor",
        type=Path,
        default=default_receptor,
        help="9GU4 receptor PDBQT prepared with AutoDockTools.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="Vina configuration from candidate docking; defaults are used if absent.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Persistent output directory.",
    )
    parser.add_argument(
        "--render-pymol",
        action="store_true",
        help="Run pymol -cq after docking to generate PNG and PSE files.",
    )
    parser.add_argument("--center-x", type=float, default=None)
    parser.add_argument("--center-y", type=float, default=None)
    parser.add_argument("--center-z", type=float, default=None)
    parser.add_argument("--size-x", type=float, default=None)
    parser.add_argument("--size-y", type=float, default=None)
    parser.add_argument("--size-z", type=float, default=None)
    parser.add_argument("--exhaustiveness", type=int, default=None)
    parser.add_argument("--num-modes", type=int, default=None)
    parser.add_argument("--energy-range", type=float, default=None)
    parser.add_argument("--cpu", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ph", type=float, default=DEFAULT_PH)
    parser.add_argument(
        "--rmsd-cutoff",
        type=float,
        default=DEFAULT_RMSD_CUTOFF,
        help="RMSD cutoff for successful redocking.",
    )
    parser.add_argument(
        "--download-timeout",
        type=int,
        default=45,
        help="RCSB download timeout in seconds.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    directories = prepare_output_directories(args.output_dir)
    run_timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    receptor = require_file(args.receptor, "receptor PDBQT")
    config_path = args.config.expanduser().resolve()
    config = parse_vina_config(config_path if config_path.is_file() else None)
    receptor_snapshot = directories["inputs"] / "NLRP3_9GU4_ADT.pdbqt"
    shutil.copy2(receptor, receptor_snapshot)
    if config_path.is_file():
        shutil.copy2(config_path, directories["inputs"] / "vina_config_used.txt")
    center = (
        resolve_parameter(args.center_x, config, "center_x", DEFAULT_CENTER[0]),
        resolve_parameter(args.center_y, config, "center_y", DEFAULT_CENTER[1]),
        resolve_parameter(args.center_z, config, "center_z", DEFAULT_CENTER[2]),
    )
    box_size = (
        resolve_parameter(args.size_x, config, "size_x", DEFAULT_BOX_SIZE[0]),
        resolve_parameter(args.size_y, config, "size_y", DEFAULT_BOX_SIZE[1]),
        resolve_parameter(args.size_z, config, "size_z", DEFAULT_BOX_SIZE[2]),
    )
    exhaustiveness = int(
        resolve_parameter(
            float(args.exhaustiveness) if args.exhaustiveness is not None else None,
            config,
            "exhaustiveness",
            float(DEFAULT_EXHAUSTIVENESS),
        )
    )
    num_modes = int(
        resolve_parameter(
            float(args.num_modes) if args.num_modes is not None else None,
            config,
            "num_modes",
            float(DEFAULT_NUM_MODES),
        )
    )
    energy_range = resolve_parameter(
        args.energy_range, config, "energy_range", DEFAULT_ENERGY_RANGE
    )
    cpu = int(
        resolve_parameter(
            float(args.cpu) if args.cpu is not None else None,
            config,
            "cpu",
            float(DEFAULT_CPU),
        )
    )
    seed = int(
        resolve_parameter(
            float(args.seed) if args.seed is not None else None,
            config,
            "seed",
            float(DEFAULT_SEED),
        )
    )
    if any((x <= 0 for x in box_size)):
        fail(f"Docking-box dimensions must be positive: {box_size}")
    if exhaustiveness <= 0 or num_modes <= 0 or cpu <= 0:
        fail("exhaustiveness, num_modes, and cpu must be positive integers.")
    print_header("NLRP3 9GU4 / NP3-253 redocking validation")
    print(f"Receptor PDBQT  : {receptor}")
    print(
        f"Vina config     : {(config_path if config_path.is_file() else 'not found; using script defaults')}"
    )
    print(f"Output directory: {directories['root']}")
    print(f"Native ligand   : {LIGAND_NAME} / CCD {LIGAND_CCD_ID}")
    print(f"Box center      : ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
    print(
        f"Box size        : ({box_size[0]:.3f}, {box_size[1]:.3f}, {box_size[2]:.3f}) Å"
    )
    print(f"exhaustiveness : {exhaustiveness}")
    print(f"num_modes      : {num_modes}")
    print(f"energy_range   : {energy_range}")
    print(f"CPU            : {cpu}")
    print(f"seed           : {seed}")
    print(f"Protonation pH  : {args.ph}")
    print(f"RMSD cutoff     : <= {args.rmsd_cutoff:.2f} Å")
    print("Persistent output: enabled")
    print_subheader("1. Retrieve experimental NP3-253 coordinates from 9GU4")
    sdf_text, source_url = fetch_native_ligand_sdf(timeout=args.download_timeout)
    raw_crystal_sdf = directories["inputs"] / "NP3_253_crystal_RCSB.sdf"
    raw_crystal_sdf.write_text(sdf_text, encoding="utf-8")
    crystal_mol = load_first_valid_sdf_molecule(sdf_text)
    crystal_sdf = directories["inputs"] / "NP3_253_crystal_normalized.sdf"
    crystal_pdb = directories["visualization"] / "NP3_253_crystal.pdb"
    write_rdkit_sdf(crystal_mol, crystal_sdf)
    from rdkit import Chem

    Chem.MolToPDBFile(crystal_mol, str(crystal_pdb))
    require_file(crystal_pdb, "crystal-ligand PDB")
    crystal_heavy = sum(
        (1 for atom in crystal_mol.GetAtoms() if atom.GetAtomicNum() > 1)
    )
    print(f"[OK] RCSB source: {source_url}")
    print(f"[OK] Crystal-ligand heavy atoms: {crystal_heavy}")
    print(f"[OK] Crystal conformers: {crystal_mol.GetNumConformers()}")
    print_subheader(
        "2. Prepare NP3-253 while preserving experimental heavy-atom coordinates"
    )
    prepared_mol = protonate_with_openbabel_in_memory(crystal_mol, args.ph)
    ph_tag = format(args.ph, "g").replace(".", "_")
    prepared_sdf = directories["inputs"] / f"NP3_253_prepared_pH{ph_tag}.sdf"
    write_rdkit_sdf(prepared_mol, prepared_sdf)
    total_atoms = prepared_mol.GetNumAtoms()
    heavy_atoms = sum(
        (1 for atom in prepared_mol.GetAtoms() if atom.GetAtomicNum() > 1)
    )
    formal_charge = sum((atom.GetFormalCharge() for atom in prepared_mol.GetAtoms()))
    print(f"[OK] Prepared atom counts: total={total_atoms}, heavy={heavy_atoms}")
    print(f"[OK] RDKit formal charge: {formal_charge:+d}")
    ligand_pdbqt = prepare_ligand_pdbqt_in_memory(prepared_mol)
    if "ROOT" not in ligand_pdbqt or "TORSDOF" not in ligand_pdbqt:
        fail("Generated NP3-253 PDBQT lacks ROOT/TORSDOF records.")
    torsdof_match = re.search("^TORSDOF\\s+(\\d+)", ligand_pdbqt, re.MULTILINE)
    torsdof = int(torsdof_match.group(1)) if torsdof_match else -1
    prepared_pdbqt = directories["inputs"] / "NP3_253_prepared.pdbqt"
    prepared_pdbqt.write_text(ligand_pdbqt, encoding="utf-8")
    print(f"[OK] Meeko generated the ligand PDBQT in memory; TORSDOF={torsdof}")
    print_subheader("3. Run Vina redocking")
    poses_pdbqt, affinities, backend = run_vina(
        receptor=receptor,
        ligand_pdbqt=ligand_pdbqt,
        center=center,
        box_size=box_size,
        exhaustiveness=exhaustiveness,
        num_modes=num_modes,
        energy_range=energy_range,
        cpu=cpu,
        seed=seed,
    )
    if not poses_pdbqt.strip():
        fail("Vina returned no poses.")
    all_poses_pdbqt = directories["docking"] / "NP3_253_redocked_all_poses.pdbqt"
    all_poses_pdbqt.write_text(poses_pdbqt, encoding="utf-8")
    if not affinities:
        affinities = parse_vina_affinities_from_pdbqt(poses_pdbqt)
    print(f"[OK] Docking backend: {backend}")
    print(f"[OK] Vina scores returned: {len(affinities)}")
    print_subheader(
        "4. Calculate symmetry-corrected heavy-atom RMSD in the fixed receptor frame"
    )
    docked_mol = docked_pdbqt_to_rdkit_mol(poses_pdbqt)
    rmsds, mapped_heavy_atoms = symmetry_corrected_fixed_frame_rmsds(
        crystal_mol, docked_mol
    )
    if not rmsds:
        fail("No redocked pose is available for RMSD calculation.")
    n = min(len(rmsds), len(affinities)) if affinities else len(rmsds)
    if n == 0:
        fail("Both Vina scores and RMSD results are empty.")
    if len(rmsds) != len(affinities):
        print(
            f"[WARN] Pose and score counts differ: poses={len(rmsds)}, scores={len(affinities)}; reporting the first {n} matched results."
        )
    print(f"[OK] Heavy atoms mapped for RMSD: {mapped_heavy_atoms}")
    print()
    print(f"{'Pose':>5}  {'Affinity (kcal/mol)':>20}  {'RMSD (Å)':>10}  {'≤cutoff':>9}")
    print(f"{'-' * 5}  {'-' * 20}  {'-' * 10}  {'-' * 9}")
    rows: List[Tuple[int, Optional[float], float]] = []
    for i in range(n):
        affinity = affinities[i] if i < len(affinities) else None
        rmsd = rmsds[i]
        passed = "YES" if rmsd <= args.rmsd_cutoff else "NO"
        rows.append((i + 1, affinity, rmsd))
        print(
            f"{i + 1:>5d}  {format_float(affinity, 3):>20}  {rmsd:>10.3f}  {passed:>9}"
        )
    top1_pose, top1_affinity, top1_rmsd = rows[0]
    lowest_rmsd_pose, lowest_rmsd_affinity, lowest_rmsd = min(
        rows, key=lambda row: row[2]
    )
    best_score_pose, best_score_affinity, best_score_rmsd = min(
        rows, key=lambda row: math.inf if row[1] is None else row[1]
    )
    print_header("Redocking conclusion")
    print(
        f"Top-ranked Vina pose: affinity={format_float(top1_affinity)} kcal/mol，RMSD={top1_rmsd:.3f} Å"
    )
    print(
        f"Lowest-RMSD pose: pose={lowest_rmsd_pose}，affinity={format_float(lowest_rmsd_affinity)} kcal/mol，RMSD={lowest_rmsd:.3f} Å"
    )
    top1_pass = top1_rmsd <= args.rmsd_cutoff
    any_pass = lowest_rmsd <= args.rmsd_cutoff
    if top1_pass:
        verdict = "PASS: the top-ranked Vina pose reproduces the crystal pose; both search and ranking succeeded."
    elif any_pass:
        verdict = "PARTIAL PASS: Vina found a crystal-like pose but did not rank it first; interpret ranking cautiously."
    else:
        verdict = "FAIL: no output pose meets the RMSD cutoff; inspect protonation, box definition, receptor preparation, or search effort."
    print(f"Verdict         : {verdict}")
    print(
        "Interpretation  : redocking validates pose recovery; a passing result does not make a Vina score equivalent to experimental Kd or binding free energy."
    )
    print_subheader("5. Save output files")
    pose_files = save_docked_pose_files(
        docked_mol=docked_mol,
        affinities=affinities,
        rmsds=rmsds,
        directories=directories,
        top1_pose=top1_pose,
        lowest_rmsd_pose=lowest_rmsd_pose,
    )
    protein_visualization = find_and_copy_protein_for_visualization(
        receptor=receptor, visualization_dir=directories["visualization"]
    )
    pymol_files = create_pymol_overlay_script(
        protein_path=protein_visualization,
        crystal_pdb=crystal_pdb,
        top1_pdb=pose_files["top1_pdb"],
        lowest_rmsd_pdb=pose_files["lowest_rmsd_pdb"],
        pml_path=directories["visualization"] / "NP3_253_redocking_overlay.pml",
        top1_rmsd=top1_rmsd,
        lowest_rmsd=lowest_rmsd,
        lowest_rmsd_pose=lowest_rmsd_pose,
    )
    csv_path = directories["results"] / "redocking_scores_rmsd.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "pose",
                "vina_affinity_kcal_mol",
                "fixed_frame_heavy_atom_rmsd_A",
                f"rmsd_le_{args.rmsd_cutoff:g}_A",
            ]
        )
        for pose_number, affinity, rmsd in rows:
            writer.writerow(
                [
                    pose_number,
                    "" if affinity is None else f"{affinity:.6f}",
                    f"{rmsd:.6f}",
                    rmsd <= args.rmsd_cutoff,
                ]
            )
    pymol_log = directories["logs"] / "pymol_render.log"
    pymol_rendered = False
    if args.render_pymol:
        pymol_rendered = render_pymol_script(
            pml_path=pymol_files["pml"], log_path=pymol_log
        )
    report_lines = [
        "NLRP3 9GU4 / NP3-253 redocking results",
        f"Run time: {run_timestamp}",
        f"Docking backend: {backend}",
        f"Receptor: {receptor}",
        f"Configuration: {(config_path if config_path.is_file() else 'script defaults')}",
        f"Box center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) Å",
        f"Box size: ({box_size[0]:.3f}, {box_size[1]:.3f}, {box_size[2]:.3f}) Å",
        f"Vina parameters: exhaustiveness={exhaustiveness}, num_modes={num_modes}, energy_range={energy_range}, cpu={cpu}, seed={seed}",
        f"Protonation pH: {args.ph}",
        f"RMSD cutoff: <= {args.rmsd_cutoff:.3f} Å",
        f"Top 1：affinity={format_float(top1_affinity)} kcal/mol, RMSD={top1_rmsd:.3f} Å",
        f"Lowest-RMSD pose: pose={lowest_rmsd_pose}, affinity={format_float(lowest_rmsd_affinity)} kcal/mol, RMSD={lowest_rmsd:.3f} Å",
        f"Verdict: {verdict}",
    ]
    report_text = "\n".join(report_lines) + "\n"
    text_summary = directories["results"] / "redocking_summary.txt"
    run_log = directories["logs"] / "redocking_run.log"
    text_summary.write_text(report_text, encoding="utf-8")
    run_log.write_text(
        report_text
        + f"Command: {' '.join(sys.argv)}\n"
        + f"Output directory: {directories['root']}\n",
        encoding="utf-8",
    )
    json_summary = directories["results"] / "redocking_summary.json"
    summary_data = {
        "run_timestamp": run_timestamp,
        "target": "NLRP3",
        "pdb_id": PDB_ID,
        "ligand": LIGAND_NAME,
        "ligand_ccd_id": LIGAND_CCD_ID,
        "crystal_ligand_source_url": source_url,
        "backend": backend,
        "command": sys.argv,
        "input": {
            "receptor": str(receptor),
            "config": str(config_path) if config_path.is_file() else None,
        },
        "parameters": {
            "center_A": list(center),
            "box_size_A": list(box_size),
            "exhaustiveness": exhaustiveness,
            "num_modes": num_modes,
            "energy_range_kcal_mol": energy_range,
            "cpu": cpu,
            "seed": seed,
            "ph": args.ph,
            "rmsd_cutoff_A": args.rmsd_cutoff,
        },
        "atom_counts": {
            "crystal_heavy_atoms": crystal_heavy,
            "prepared_total_atoms": total_atoms,
            "prepared_heavy_atoms": heavy_atoms,
            "rmsd_mapped_heavy_atoms": mapped_heavy_atoms,
            "formal_charge": formal_charge,
            "torsdof": torsdof,
        },
        "top1": {
            "pose": top1_pose,
            "affinity_kcal_mol": top1_affinity,
            "fixed_frame_heavy_atom_rmsd_A": top1_rmsd,
            "passes_cutoff": top1_pass,
        },
        "lowest_rmsd": {
            "pose": lowest_rmsd_pose,
            "affinity_kcal_mol": lowest_rmsd_affinity,
            "fixed_frame_heavy_atom_rmsd_A": lowest_rmsd,
            "passes_cutoff": any_pass,
        },
        "best_vina_score": {
            "pose": best_score_pose,
            "affinity_kcal_mol": best_score_affinity,
            "fixed_frame_heavy_atom_rmsd_A": best_score_rmsd,
        },
        "verdict": verdict,
        "poses": [
            {
                "pose": pose_number,
                "affinity_kcal_mol": affinity,
                "fixed_frame_heavy_atom_rmsd_A": rmsd,
                "passes_cutoff": rmsd <= args.rmsd_cutoff,
            }
            for pose_number, affinity, rmsd in rows
        ],
        "pymol_render_requested": args.render_pymol,
        "pymol_render_succeeded": pymol_rendered,
        "files": {
            "raw_crystal_sdf": str(raw_crystal_sdf),
            "normalized_crystal_sdf": str(crystal_sdf),
            "prepared_ligand_sdf": str(prepared_sdf),
            "prepared_ligand_pdbqt": str(prepared_pdbqt),
            "all_poses_pdbqt": str(all_poses_pdbqt),
            **{key: str(value) for key, value in pose_files.items()},
            "scores_csv": str(csv_path),
            "text_summary": str(text_summary),
            "pymol_script": str(pymol_files["pml"]),
            "pymol_top1_png": str(pymol_files["top1_png"]) if pymol_rendered else None,
            "pymol_lowest_rmsd_png": (
                str(pymol_files["lowest_rmsd_png"]) if pymol_rendered else None
            ),
            "pymol_session": str(pymol_files["pse"]) if pymol_rendered else None,
            "run_log": str(run_log),
        },
    }
    json_summary.write_text(
        json.dumps(summary_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[OK] All Vina poses: {all_poses_pdbqt}")
    print(f"[OK] Score/RMSD table: {csv_path}")
    print(f"[OK] JSON summary: {json_summary}")
    print(f"[OK] PyMOL overlay script: {pymol_files['pml']}")
    if pymol_rendered:
        print(f"[OK] PyMOL image: {pymol_files['top1_png']}")
        print(f"[OK] PyMOL session: {pymol_files['pse']}")
    elif not args.render_pymol:
        print(
            f"[INFO] PyMOL rendering was not requested. Run when needed: pymol {pymol_files['pml']}"
        )
    print(f"Files saved      : {directories['root']}")


if __name__ == "__main__":
    main()
