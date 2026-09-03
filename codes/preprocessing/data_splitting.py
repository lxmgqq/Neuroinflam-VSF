#!/usr/bin/env python3
"""Create reproducible dataset splits from preprocessed inputs.

B3DB and BBBP use Bemis-Murcko scaffold splits. Davis, KIBA, and BindingDB use
ligand scaffold clusters and CD-HIT protein clusters for warm, cold-drug,
cold-target, and double-cold evaluation. BindingDB proteins with at least 40%
global sequence identity to human NLRP3 are excluded with CD-HIT-2D.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import numpy as np
    import pandas as pd
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]
    pd = None  # type: ignore[assignment]

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:  # pragma: no cover
    Chem = None  # type: ignore[assignment]
    RDLogger = None  # type: ignore[assignment]
    MurckoScaffold = None  # type: ignore[assignment]


DATASET_NAMES = ("B3DB", "BBBP", "Davis", "KIBA", "BindingDB")
CLASSIFICATION_DATASETS = frozenset({"B3DB", "BBBP"})
DTA_DATASETS = frozenset({"Davis", "KIBA", "BindingDB"})
SCENARIOS = ("warm", "cold_drug", "cold_target", "double_cold")

DEFAULT_DATA_ROOT = Path("./data/datasets")
DEFAULT_NLRP3_FASTA = Path("./data/datasets/BindingDB/NLRP3_Q96P20.fasta")
NLRP3_ACCESSION = "Q96P20"
NLRP3_EXPECTED_LENGTH = 1036
NLRP3_FASTA_URL = (
    f"https://rest.uniprot.org/uniprotkb/{NLRP3_ACCESSION}.fasta"
)

CDHIT_IDENTITY = 0.40
CDHIT_WORD_SIZE = 2
CDHIT_THREADS = 16
CDHIT_MEMORY_MB = 0

CLASSIFICATION_RATIOS = (0.70, 0.10, 0.20)
DAVIS_KIBA_RATIOS = (0.80, 0.10, 0.10)
BINDINGDB_RATIOS = (0.70, 0.10, 0.20)

CLASSIFICATION_FIXED_TEST_SEED = 2025
WARM_TEST_SEED = 2025
COLD_DRUG_TEST_SEED = 2026
COLD_TARGET_TEST_SEED = 2027
DOUBLE_COLD_TEST_DRUG_SEED = 2028
DOUBLE_COLD_TEST_TARGET_SEED = 2029


@dataclass(frozen=True)
class DatasetPaths:
    name: str
    dataset_dir: Path
    inputs: tuple[Path, ...]
    output_dir: Path


def require_dependencies() -> None:
    missing = []
    if np is None:
        missing.append("numpy")
    if pd is None:
        missing.append("pandas")
    if Chem is None or MurckoScaffold is None:
        missing.append("rdkit")
    if missing:
        raise RuntimeError("Missing splitting dependencies: " + ", ".join(missing))


def parse_seed_spec(text: str) -> list[int]:
    seeds: list[int] = []
    for token in (part.strip() for part in text.split(",")):
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"Invalid seed range: {token}")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(token))
    seeds = list(dict.fromkeys(seeds))
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("Seed lists must contain non-negative integers")
    return seeds


def normalize_dataset_names(values: Sequence[str]) -> list[str]:
    if not values or any(value.casefold() == "all" for value in values):
        return list(DATASET_NAMES)
    real = {name.casefold(): name for name in DATASET_NAMES}
    selected = []
    for value in values:
        key = value.casefold()
        if key not in real:
            raise ValueError(f"Unknown dataset {value!r}; choose from {DATASET_NAMES}")
        if real[key] not in selected:
            selected.append(real[key])
    return selected


def build_dataset_paths(data_root: Path, name: str) -> DatasetPaths:
    dataset_dir = data_root / name
    if name == "B3DB":
        inputs = (dataset_dir / "b3db_preprocessed.csv",)
    elif name == "BBBP":
        inputs = (dataset_dir / "bbbp_preprocessed.csv",)
    elif name in {"Davis", "KIBA"}:
        inputs = (
            dataset_dir / "ligands_preprocessed.csv",
            dataset_dir / "proteins_preprocessed.csv",
            dataset_dir / "interactions_preprocessed.csv",
        )
    elif name == "BindingDB":
        inputs = (dataset_dir / "bindingdb_preprocessed.db",)
    else:  # pragma: no cover
        raise ValueError(name)
    return DatasetPaths(name, dataset_dir, inputs, dataset_dir / "splits")


def validate_ratios(ratios: tuple[float, float, float], dataset: str) -> None:
    if any(value <= 0 for value in ratios) or not math.isclose(sum(ratios), 1.0):
        raise ValueError(f"Invalid train/val/test ratios for {dataset}: {ratios}")


def require_nonempty_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"Empty {label}: {path}")


def require_columns(frame: Any, required: set[str], source: Path) -> None:
    missing = required - set(str(column) for column in frame.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")


def validate_classification_input(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    require_columns(frame, {"SMILES", "label"}, path)
    if frame.empty:
        raise ValueError(f"No rows in {path}")
    if frame[["SMILES", "label"]].isna().any().any():
        raise ValueError(f"Preprocessed input unexpectedly contains missing values: {path}")
    labels = set(pd.to_numeric(frame["label"], errors="raise").astype(int).unique())
    if not labels <= {0, 1}:
        raise ValueError(f"Preprocessed labels are not binary in {path}: {sorted(labels)}")
    if frame["SMILES"].astype(str).duplicated().any():
        raise ValueError(f"Preprocessed SMILES are not unique in {path}")
    return {"rows": int(len(frame)), "labels": sorted(int(value) for value in labels)}


def validate_dta_csv_inputs(paths: DatasetPaths) -> dict[str, Any]:
    ligand_path, protein_path, interaction_path = paths.inputs
    ligands = pd.read_csv(ligand_path)
    proteins = pd.read_csv(protein_path)
    interactions = pd.read_csv(interaction_path)
    require_columns(ligands, {"drug_id", "SMILES"}, ligand_path)
    require_columns(proteins, {"target_id", "Protein Sequence"}, protein_path)
    require_columns(interactions, {"drug_id", "target_id", "affinity"}, interaction_path)
    if ligands.empty or proteins.empty or interactions.empty:
        raise ValueError(f"Preprocessed {paths.name} entity/interaction files must be non-empty")
    if ligands["drug_id"].astype(str).duplicated().any():
        raise ValueError(f"Duplicate drug_id values in {ligand_path}")
    if proteins["target_id"].astype(str).duplicated().any():
        raise ValueError(f"Duplicate target_id values in {protein_path}")
    drug_ids = set(ligands["drug_id"].astype(str))
    target_ids = set(proteins["target_id"].astype(str))
    missing_drugs = set(interactions["drug_id"].astype(str)) - drug_ids
    missing_targets = set(interactions["target_id"].astype(str)) - target_ids
    if missing_drugs or missing_targets:
        raise ValueError(
            f"{paths.name} interactions reference missing entities: "
            f"drugs={list(sorted(missing_drugs))[:5]}, targets={list(sorted(missing_targets))[:5]}"
        )
    affinity = pd.to_numeric(interactions["affinity"], errors="raise").to_numpy(float)
    if not np.isfinite(affinity).all():
        raise ValueError(f"Non-finite affinity values in {interaction_path}")
    return {
        "ligands": int(len(ligands)),
        "proteins": int(len(proteins)),
        "interactions": int(len(interactions)),
    }


def validate_bindingdb_input(path: Path) -> dict[str, Any]:
    required = {
        "BindingDB MonomerID",
        "Ligand SMILES",
        "BindingDB Target Chain Sequence",
        "Sequence Hash",
        "Kd (nM)",
    }
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        columns = {
            str(row[1]) for row in conn.execute('PRAGMA table_info("binding_data")')
        }
        missing = required - columns
        if missing:
            raise ValueError(f"BindingDB binding_data is missing columns: {sorted(missing)}")
        rows = int(conn.execute('SELECT COUNT(*) FROM "binding_data"').fetchone()[0])
        if rows <= 0:
            raise ValueError("BindingDB binding_data contains no rows")
        ligand_columns = {
            str(row[1]) for row in conn.execute('PRAGMA table_info("ligand")')
        }
        protein_columns = {
            str(row[1]) for row in conn.execute('PRAGMA table_info("protein")')
        }
        if {"BindingDB MonomerID", "Ligand SMILES"} - ligand_columns:
            raise ValueError("BindingDB ligand lookup table is missing required columns")
        if {"Sequence Hash", "BindingDB Target Chain Sequence"} - protein_columns:
            raise ValueError("BindingDB protein lookup table is missing required columns")
    finally:
        conn.close()
    return {"binding_rows": rows}


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks).upper()))
                header, chunks = line[1:].split()[0], []
            else:
                if header is None:
                    raise ValueError(f"Sequence before FASTA header in {path}")
                chunks.append("".join(line.split()))
    if header is not None:
        records.append((header, "".join(chunks).upper()))
    if not records or any(not sequence for _, sequence in records):
        raise ValueError(f"Invalid or empty FASTA: {path}")
    return records


def validate_canonical_nlrp3(
    records: Sequence[tuple[str, str]], source: Path
) -> tuple[str, str]:
    if len(records) != 1:
        raise ValueError(f"NLRP3 FASTA must contain exactly one sequence: {source}")
    header, sequence = records[0]
    invalid = sorted(set(sequence) - set("ACDEFGHIKLMNPQRSTVWYBXZOUJ"))
    if invalid:
        raise ValueError(f"Invalid NLRP3 sequence letters in {source}: {invalid}")
    if NLRP3_ACCESSION not in header or len(sequence) != NLRP3_EXPECTED_LENGTH:
        raise ValueError(
            "Downloaded FASTA does not match canonical human NLRP3 "
            f"{NLRP3_ACCESSION}: header={header!r}, length={len(sequence)}"
        )
    return header, sequence


def ensure_nlrp3_fasta(path: Path) -> bool:
    if path.is_file() and path.stat().st_size > 0:
        return False

    print(f"[NLRP3] Downloading canonical FASTA from UniProt: {NLRP3_FASTA_URL}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    request = urllib.request.Request(
        NLRP3_FASTA_URL,
        headers={"User-Agent": "Neuroinflam-VSF/1.0"},
    )
    try:
        try:
            import certifi
        except ImportError:
            tls_context = ssl.create_default_context()
        else:
            tls_context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(
            request, timeout=60, context=tls_context
        ) as response:
            payload = response.read().decode("utf-8")
        temporary.write_text(payload.rstrip() + "\n", encoding="utf-8", newline="\n")
        validate_canonical_nlrp3(read_fasta(temporary), temporary)
        os.replace(temporary, path)
    except (urllib.error.URLError, TimeoutError, OSError, UnicodeError) as exc:
        raise RuntimeError(
            f"Could not download the NLRP3 FASTA from {NLRP3_FASTA_URL}: {exc}"
        ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"[NLRP3] Saved canonical FASTA: {path}")
    return True


def find_executable(name: str) -> str:
    resolved = shutil.which(name)
    if resolved is None:
        raise FileNotFoundError(
            f"Cannot find {name!r}. Install CD-HIT, for example: "
            "conda install -c bioconda cd-hit"
        )
    return resolved


def run_preflight(
    selected: Sequence[str],
    data_root: Path,
    nlrp3_fasta: Path,
    overwrite: bool,
    check_outputs: bool = True,
) -> dict[str, Any]:
    file_problems: list[str] = []
    for name in selected:
        paths = build_dataset_paths(data_root, name)
        for input_path in paths.inputs:
            if not input_path.is_file():
                file_problems.append(f"missing: {name} preprocessed input: {input_path}")
            elif input_path.stat().st_size <= 0:
                file_problems.append(f"empty: {name} preprocessed input: {input_path}")
        if check_outputs and paths.output_dir.exists() and not overwrite:
            file_problems.append(
                f"output exists: {paths.output_dir} (pass --overwrite to replace it)"
            )
    if "BindingDB" in selected:
        ensure_nlrp3_fasta(nlrp3_fasta)
    if file_problems:
        raise FileNotFoundError("Splitting preflight failed:\n- " + "\n- ".join(file_problems))

    require_dependencies()
    report: dict[str, Any] = {}
    needs_cdhit = any(name in DTA_DATASETS for name in selected)
    if needs_cdhit:
        report["cd-hit"] = find_executable("cd-hit")
    if "BindingDB" in selected:
        report["cd-hit-2d"] = find_executable("cd-hit-2d")
        records = read_fasta(nlrp3_fasta)
        header, sequence = validate_canonical_nlrp3(records, nlrp3_fasta)
        report["NLRP3"] = {
            "path": str(nlrp3_fasta),
            "source_url": NLRP3_FASTA_URL,
            "header": header,
            "length": len(sequence),
            "sha256": hashlib.sha256(sequence.encode()).hexdigest(),
        }

    for name in selected:
        paths = build_dataset_paths(data_root, name)
        if name in CLASSIFICATION_DATASETS:
            report[name] = validate_classification_input(paths.inputs[0])
        elif name in {"Davis", "KIBA"}:
            report[name] = validate_dta_csv_inputs(paths)
        else:
            report[name] = validate_bindingdb_input(paths.inputs[0])
    return report


def scaffold_key(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES found after preprocessing: {smiles!r}")
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=True
        )
    except ValueError:
        # Some valid charged structures cannot be sanitized after Murcko side-chain
        # removal. Keep each such structure as its own scaffold instead of aborting
        # the entire split-generation run.
        scaffold = ""
    if not scaffold:
        scaffold = str(smiles)
    return scaffold


def stable_cluster_id(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]}"


def add_ligand_clusters(ligands: Any) -> Any:
    result = ligands.copy()
    result["scaffold"] = result["SMILES"].astype(str).map(scaffold_key)
    result["ligand_cluster_id"] = result["scaffold"].map(
        lambda value: stable_cluster_id("scaf", value)
    )
    return result


def select_clusters_for_sample_target(
    cluster_sizes: Any, target_count: int, seed: int
) -> set[str]:
    cluster_ids = sorted(str(value) for value in cluster_sizes.index)
    rng = np.random.RandomState(seed)
    rng.shuffle(cluster_ids)
    selected: list[str] = []
    count = 0
    for cluster_id in cluster_ids:
        size = int(cluster_sizes.loc[cluster_id])
        if count + size < target_count:
            selected.append(cluster_id)
            count += size
            continue
        if not selected or abs((count + size) - target_count) <= abs(count - target_count):
            selected.append(cluster_id)
        break
    if not selected or len(selected) >= len(cluster_ids):
        raise ValueError("Too few scaffold clusters for non-empty train/validation/test splits")
    return set(selected)


def save_csv(frame: Any, path: Path, columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, list(columns)].to_csv(path, index=False)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def classification_split(
    dataset_name: str,
    input_path: Path,
    output_dir: Path,
    seeds: Sequence[int],
) -> dict[str, Any]:
    train_ratio, val_ratio, test_ratio = CLASSIFICATION_RATIOS
    frame = pd.read_csv(input_path)[["SMILES", "label"]].copy()
    frame["SMILES"] = frame["SMILES"].astype(str)
    frame["label"] = frame["label"].astype(int)
    frame["scaffold"] = frame["SMILES"].map(scaffold_key)
    frame["scaffold_cluster_id"] = frame["scaffold"].map(
        lambda value: stable_cluster_id("scaf", value)
    )
    cluster_sizes = frame.groupby("scaffold_cluster_id").size()
    test_clusters = select_clusters_for_sample_target(
        cluster_sizes,
        max(1, round(len(frame) * test_ratio)),
        CLASSIFICATION_FIXED_TEST_SEED,
    )
    pool = frame[~frame["scaffold_cluster_id"].isin(test_clusters)].copy()
    test = frame[frame["scaffold_cluster_id"].isin(test_clusters)].copy()
    pool_sizes = pool.groupby("scaffold_cluster_id").size()
    if test.empty or pool.empty:
        raise ValueError(f"{dataset_name} scaffold test split is empty")

    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "scaffold_clusters.csv", index=False)
    seed_reports: dict[str, Any] = {}
    for seed in seeds:
        val_clusters = select_clusters_for_sample_target(
            pool_sizes, max(1, round(len(frame) * val_ratio)), seed
        )
        train = pool[~pool["scaffold_cluster_id"].isin(val_clusters)].copy()
        val = pool[pool["scaffold_cluster_id"].isin(val_clusters)].copy()
        if train.empty or val.empty:
            raise ValueError(f"{dataset_name} seed {seed} has an empty split")
        train["Split"], val["Split"], test["Split"] = "train", "val", "test"

        train_clusters = set(train["scaffold_cluster_id"])
        observed_val = set(val["scaffold_cluster_id"])
        observed_test = set(test["scaffold_cluster_id"])
        if train_clusters & observed_val or train_clusters & observed_test or observed_val & observed_test:
            raise RuntimeError(f"Scaffold leakage detected in {dataset_name} seed {seed}")

        scene_dir = output_dir / f"seed_{seed:04d}" / "scaffold"
        columns = ("SMILES", "label", "Split")
        save_csv(train, scene_dir / "train.csv", columns)
        save_csv(val, scene_dir / "val.csv", columns)
        save_csv(test, scene_dir / "test.csv", columns)
        seed_reports[str(seed)] = {
            "train": int(len(train)),
            "val": int(len(val)),
            "test": int(len(test)),
            "label_distribution": {
                split: {
                    str(key): int(value)
                    for key, value in part["label"].value_counts().sort_index().items()
                }
                for split, part in (("train", train), ("val", val), ("test", test))
            },
        }

    report = {
        "dataset": dataset_name,
        "method": "Bemis-Murcko scaffold-disjoint split",
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "fixed_test_seed": CLASSIFICATION_FIXED_TEST_SEED,
        "seeds": list(seeds),
        "samples": int(len(frame)),
        "scaffold_clusters": int(frame["scaffold_cluster_id"].nunique()),
        "seed_stats": seed_reports,
    }
    write_json(output_dir / "split_manifest.json", report)
    return report


def write_fasta(records: Iterable[tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for header, sequence in records:
            stream.write(f">{header}\n")
            for start in range(0, len(sequence), 80):
                stream.write(sequence[start : start + 80] + "\n")


def parse_cdhit_clusters(path: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    cluster_id: int | None = None
    pattern = re.compile(r">([^\.\s]+)\.\.\.")
    with path.open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if line.startswith(">Cluster"):
                cluster_id = int(line.split()[1])
                continue
            match = pattern.search(line)
            if match and cluster_id is not None:
                mapping[match.group(1)] = cluster_id
    if not mapping:
        raise ValueError(f"No clusters parsed from {path}")
    return mapping


def run_command(command: Sequence[str], label: str) -> None:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} failed with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout[-4000:]}\n"
            f"stderr:\n{completed.stderr[-4000:]}"
        )


def cluster_proteins_with_cdhit(
    proteins: Any,
    work_dir: Path,
    cdhit_executable: str,
    threads: int,
) -> Any:
    proteins = proteins.sort_values("target_id").reset_index(drop=True).copy()
    headers = [f"protein_{index:08d}" for index in range(len(proteins))]
    fasta_path = work_dir / "proteins.fasta"
    output_path = work_dir / "proteins_cdhit_c0.40.fasta"
    write_fasta(zip(headers, proteins["Protein Sequence"].astype(str)), fasta_path)
    command = [
        cdhit_executable,
        "-i", str(fasta_path),
        "-o", str(output_path),
        "-c", str(CDHIT_IDENTITY),
        "-n", str(CDHIT_WORD_SIZE),
        "-G", "1",
        "-g", "1",
        "-p", "1",
        "-d", "0",
        "-M", str(CDHIT_MEMORY_MB),
        "-T", str(threads),
    ]
    run_command(command, "CD-HIT protein clustering")
    clusters = parse_cdhit_clusters(Path(str(output_path) + ".clstr"))
    missing = set(headers) - set(clusters)
    if missing:
        raise RuntimeError(f"CD-HIT omitted {len(missing)} proteins; examples={sorted(missing)[:5]}")
    proteins["protein_cluster_id"] = [f"cdhit_{clusters[header]:08d}" for header in headers]
    proteins["cdhit_identity_threshold"] = CDHIT_IDENTITY
    return proteins


def parse_cdhit2d_identities(path: Path) -> dict[str, float | None]:
    identities: dict[str, float | None] = {}
    header_pattern = re.compile(r">([^\.\s]+)\.\.\.")
    identity_pattern = re.compile(r"at\s+([0-9.]+)%")
    if not path.is_file():
        return identities
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            header_match = header_pattern.search(line)
            if not header_match or not header_match.group(1).startswith("protein_"):
                continue
            identity_match = identity_pattern.search(line)
            identities[header_match.group(1)] = (
                float(identity_match.group(1)) / 100.0 if identity_match else None
            )
    return identities


def exclude_nlrp3_similar_proteins(
    proteins: Any,
    nlrp3_fasta: Path,
    work_dir: Path,
    cdhit2d_executable: str,
    threads: int,
) -> tuple[Any, Any]:
    proteins = proteins.sort_values("target_id").reset_index(drop=True).copy()
    headers = [f"protein_{index:08d}" for index in range(len(proteins))]
    query_fasta = work_dir / "bindingdb_proteins.fasta"
    novel_fasta = work_dir / "bindingdb_proteins_novel_to_nlrp3.fasta"
    write_fasta(zip(headers, proteins["Protein Sequence"].astype(str)), query_fasta)
    command = [
        cdhit2d_executable,
        "-i", str(nlrp3_fasta),
        "-i2", str(query_fasta),
        "-o", str(novel_fasta),
        "-c", str(CDHIT_IDENTITY),
        "-n", str(CDHIT_WORD_SIZE),
        "-G", "1",
        "-g", "1",
        "-p", "1",
        "-d", "0",
        "-s2", "0.0",
        "-M", str(CDHIT_MEMORY_MB),
        "-T", str(threads),
    ]
    run_command(command, "CD-HIT-2D NLRP3 exclusion")
    novel_headers = {header for header, _ in read_fasta(novel_fasta)}
    excluded_headers = set(headers) - novel_headers
    identities = parse_cdhit2d_identities(Path(str(novel_fasta) + ".clstr"))
    header_to_index = {header: index for index, header in enumerate(headers)}
    excluded_indices = sorted(header_to_index[header] for header in excluded_headers)
    excluded = proteins.iloc[excluded_indices].copy()
    excluded["nlrp3_identity"] = [identities.get(headers[index]) for index in excluded_indices]
    excluded["identity_threshold"] = CDHIT_IDENTITY
    retained = proteins.drop(index=excluded_indices).reset_index(drop=True)
    if retained.empty:
        raise RuntimeError("NLRP3 filtering removed every BindingDB protein")
    excluded.to_csv(work_dir / "nlrp3_excluded_proteins.csv", index=False)
    return retained, excluded


def load_dta_csv_data(paths: DatasetPaths) -> tuple[Any, Any, Any]:
    ligands = pd.read_csv(paths.inputs[0])[["drug_id", "SMILES"]].copy()
    proteins = pd.read_csv(paths.inputs[1])[["target_id", "Protein Sequence"]].copy()
    interactions = pd.read_csv(paths.inputs[2])[["drug_id", "target_id", "affinity"]].copy()
    for frame, column in ((ligands, "drug_id"), (proteins, "target_id")):
        frame[column] = frame[column].astype(str)
    interactions["drug_id"] = interactions["drug_id"].astype(str)
    interactions["target_id"] = interactions["target_id"].astype(str)
    interactions["affinity"] = interactions["affinity"].astype(float)
    return ligands, proteins, interactions


def load_bindingdb_data(db_path: Path) -> tuple[Any, Any, Any, dict[str, Any]]:
    interaction_query = """
        SELECT
            "BindingDB MonomerID" AS drug_id,
            "Sequence Hash" AS target_id,
            "Kd (nM)" AS kd_nm
        FROM "binding_data"
    """
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        raw = pd.read_sql_query(interaction_query, conn)
        ligands = pd.read_sql_query(
            'SELECT "BindingDB MonomerID" AS drug_id, "Ligand SMILES" AS SMILES FROM "ligand"',
            conn,
        )
        proteins = pd.read_sql_query(
            'SELECT "Sequence Hash" AS target_id, '
            '"BindingDB Target Chain Sequence" AS "Protein Sequence" FROM "protein"',
            conn,
        )
    finally:
        conn.close()
    raw["drug_id"] = raw["drug_id"].astype(str)
    raw["target_id"] = raw["target_id"].astype(str)
    raw["kd_nm"] = raw["kd_nm"].astype(float)
    raw["affinity"] = 9.0 - np.log10(raw["kd_nm"].to_numpy(float))
    before_aggregation = len(raw)
    interactions = (
        raw.groupby(["drug_id", "target_id"], as_index=False)["affinity"]
        .mean()
        .sort_values(["drug_id", "target_id"])
        .reset_index(drop=True)
    )
    ligands["drug_id"] = ligands["drug_id"].astype(str)
    proteins["target_id"] = proteins["target_id"].astype(str)
    report = {
        "binding_rows": int(before_aggregation),
        "unique_pairs": int(len(interactions)),
        "duplicate_measurements_aggregated": int(before_aggregation - len(interactions)),
        "target": "pKd",
        "conversion": "pKd = 9 - log10(Kd_nM)",
    }
    return ligands, proteins, interactions, report


def split_entity_values(
    values: Sequence[str],
    test_ratio: float,
    train_ratio: float,
    val_ratio: float,
    test_seed: int,
    train_val_seed: int,
    label: str,
) -> tuple[set[str], set[str], set[str]]:
    ordered = np.asarray(sorted(set(str(value) for value in values)), dtype=object)
    if len(ordered) < 3:
        raise ValueError(f"At least three {label} clusters are required")
    test_rng = np.random.RandomState(test_seed)
    test_rng.shuffle(ordered)
    n_test = max(1, int(math.ceil(len(ordered) * test_ratio)))
    if n_test >= len(ordered) - 1:
        raise ValueError(f"Too few {label} clusters for test ratio {test_ratio}")
    test = set(str(value) for value in ordered[-n_test:])
    remaining = ordered[:-n_test].copy()
    train_val_rng = np.random.RandomState(train_val_seed)
    train_val_rng.shuffle(remaining)
    n_train = int(math.floor(len(remaining) * train_ratio / (train_ratio + val_ratio)))
    n_train = min(max(n_train, 1), len(remaining) - 1)
    train = set(str(value) for value in remaining[:n_train])
    val = set(str(value) for value in remaining[n_train:])
    if not train or not val or not test:
        raise ValueError(f"Empty {label} cluster partition")
    return train, val, test


def fixed_test_sets(frame: Any, ratios: tuple[float, float, float]) -> dict[str, Any]:
    _, _, test_ratio = ratios
    row_ids = frame["row_id"].to_numpy(int, copy=True)
    rng = np.random.RandomState(WARM_TEST_SEED)
    rng.shuffle(row_ids)
    n_test = max(1, int(math.ceil(len(row_ids) * test_ratio)))
    if n_test >= len(row_ids) - 1:
        raise ValueError("Too few DTA interactions for warm split")
    return {
        "warm_rows": set(int(value) for value in row_ids[-n_test:]),
        "drug_test_seed": COLD_DRUG_TEST_SEED,
        "target_test_seed": COLD_TARGET_TEST_SEED,
        "double_drug_test_seed": DOUBLE_COLD_TEST_DRUG_SEED,
        "double_target_test_seed": DOUBLE_COLD_TEST_TARGET_SEED,
    }


def split_warm(frame: Any, fixed: Mapping[str, Any], ratios: tuple[float, float, float], seed: int) -> Any:
    train_ratio, val_ratio, _ = ratios
    test_rows = fixed["warm_rows"]
    remaining = np.asarray(
        [int(value) for value in frame["row_id"] if int(value) not in test_rows], dtype=int
    )
    rng = np.random.RandomState(seed)
    rng.shuffle(remaining)
    n_train = int(math.floor(len(remaining) * train_ratio / (train_ratio + val_ratio)))
    n_train = min(max(n_train, 1), len(remaining) - 1)
    train_rows = set(int(value) for value in remaining[:n_train])
    result = frame.copy()
    result["Split"] = "val"
    result.loc[result["row_id"].isin(train_rows), "Split"] = "train"
    result.loc[result["row_id"].isin(test_rows), "Split"] = "test"
    return result


def split_cold_single(
    frame: Any,
    cluster_column: str,
    ratios: tuple[float, float, float],
    seed: int,
    test_seed: int,
    label: str,
) -> Any:
    train_ratio, val_ratio, test_ratio = ratios
    train, val, test = split_entity_values(
        frame[cluster_column].astype(str).tolist(),
        test_ratio, train_ratio, val_ratio, test_seed, seed, label,
    )
    result = frame.copy()
    result["Split"] = result[cluster_column].astype(str).map(
        lambda value: "train" if value in train else ("val" if value in val else "test")
    )
    return result


def split_double_cold(frame: Any, ratios: tuple[float, float, float], seed: int) -> Any:
    train_ratio, val_ratio, test_ratio = ratios
    drug_train, drug_val, drug_test = split_entity_values(
        frame["ligand_cluster_id"].astype(str).tolist(),
        test_ratio, train_ratio, val_ratio, DOUBLE_COLD_TEST_DRUG_SEED, seed, "ligand",
    )
    target_train, target_val, target_test = split_entity_values(
        frame["protein_cluster_id"].astype(str).tolist(),
        test_ratio,
        train_ratio,
        val_ratio,
        DOUBLE_COLD_TEST_TARGET_SEED,
        seed + 1_000_003,
        "protein",
    )
    result = frame.copy()
    result["Split"] = "exclude"
    pairs = (
        (drug_train, target_train, "train"),
        (drug_val, target_val, "val"),
        (drug_test, target_test, "test"),
    )
    for drug_clusters, target_clusters, split_name in pairs:
        mask = (
            result["ligand_cluster_id"].astype(str).isin(drug_clusters)
            & result["protein_cluster_id"].astype(str).isin(target_clusters)
        )
        result.loc[mask, "Split"] = split_name
    return result[result["Split"] != "exclude"].copy()


def assert_dta_split(scene: str, frame: Any) -> None:
    counts = frame["Split"].value_counts()
    if any(int(counts.get(name, 0)) <= 0 for name in ("train", "val", "test")):
        raise ValueError(f"{scene} has an empty train, validation, or test split: {counts.to_dict()}")
    train, val, test = (
        frame[frame["Split"] == name] for name in ("train", "val", "test")
    )
    pair_sets = [set(zip(part["drug_id"], part["target_id"])) for part in (train, val, test)]
    if pair_sets[0] & pair_sets[1] or pair_sets[0] & pair_sets[2] or pair_sets[1] & pair_sets[2]:
        raise RuntimeError(f"Drug-target pair leakage in {scene}")
    if scene in {"cold_drug", "double_cold"}:
        clusters = [set(part["ligand_cluster_id"]) for part in (train, val, test)]
        if clusters[0] & clusters[1] or clusters[0] & clusters[2] or clusters[1] & clusters[2]:
            raise RuntimeError(f"Ligand cluster leakage in {scene}")
    if scene in {"cold_target", "double_cold"}:
        clusters = [set(part["protein_cluster_id"]) for part in (train, val, test)]
        if clusters[0] & clusters[1] or clusters[0] & clusters[2] or clusters[1] & clusters[2]:
            raise RuntimeError(f"Protein cluster leakage in {scene}")


def create_dta_splits(
    dataset_name: str,
    ligands: Any,
    proteins: Any,
    interactions: Any,
    output_dir: Path,
    seeds: Sequence[int],
    ratios: tuple[float, float, float],
    cdhit_executable: str,
    threads: int,
    extra_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ligands = add_ligand_clusters(ligands)
    with tempfile.TemporaryDirectory(prefix="cdhit_") as work_dir:
        proteins = cluster_proteins_with_cdhit(
            proteins, Path(work_dir), cdhit_executable, threads
        )
    frame = interactions.merge(
        ligands[["drug_id", "ligand_cluster_id"]], on="drug_id", how="left", validate="many_to_one"
    ).merge(
        proteins[["target_id", "protein_cluster_id"]], on="target_id", how="left", validate="many_to_one"
    )
    if frame[["ligand_cluster_id", "protein_cluster_id"]].isna().any().any():
        raise RuntimeError(f"{dataset_name} interactions lost cluster assignments")
    frame = frame.sort_values(["drug_id", "target_id"]).reset_index(drop=True)
    frame["row_id"] = np.arange(len(frame), dtype=np.int64)
    fixed = fixed_test_sets(frame, ratios)

    output_dir.mkdir(parents=True, exist_ok=True)
    seed_reports: dict[str, Any] = {}
    for seed in seeds:
        scene_frames = {
            "warm": split_warm(frame, fixed, ratios, seed),
            "cold_drug": split_cold_single(
                frame, "ligand_cluster_id", ratios, seed, COLD_DRUG_TEST_SEED, "ligand"
            ),
            "cold_target": split_cold_single(
                frame, "protein_cluster_id", ratios, seed, COLD_TARGET_TEST_SEED, "protein"
            ),
            "double_cold": split_double_cold(frame, ratios, seed),
        }
        seed_reports[str(seed)] = {}
        for scene, scene_frame in scene_frames.items():
            assert_dta_split(scene, scene_frame)
            scene_dir = output_dir / f"seed_{seed:04d}" / scene
            columns = ("drug_id", "target_id", "affinity", "Split")
            stats: dict[str, int] = {}
            for split_name in ("train", "val", "test"):
                part = scene_frame[scene_frame["Split"] == split_name]
                save_csv(part, scene_dir / f"{split_name}.csv", columns)
                stats[split_name] = int(len(part))
            seed_reports[str(seed)][scene] = stats

    train_ratio, val_ratio, test_ratio = ratios
    report: dict[str, Any] = {
        "dataset": dataset_name,
        "ligand_clustering": "RDKit Bemis-Murcko scaffold",
        "protein_clustering": {
            "method": "CD-HIT global sequence identity",
            "identity_threshold": CDHIT_IDENTITY,
            "word_size": CDHIT_WORD_SIZE,
        },
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "seeds": list(seeds),
        "interactions": int(len(frame)),
        "ligands": int(ligands["drug_id"].nunique()),
        "proteins": int(proteins["target_id"].nunique()),
        "ligand_clusters": int(ligands["ligand_cluster_id"].nunique()),
        "protein_clusters": int(proteins["protein_cluster_id"].nunique()),
        "seed_stats": seed_reports,
    }
    if extra_report:
        report.update(dict(extra_report))
    return report


def stage_path(final: Path) -> Path:
    return final.with_name(f".{final.name}_staging_{os.getpid()}")


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def publish_outputs(staged_to_final: Mapping[Path, Path], overwrite: bool) -> None:
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for staged, final in staged_to_final.items():
            if not staged.exists():
                raise FileNotFoundError(f"Missing staged split output: {staged}")
            backup = final.with_name(f".{final.name}_backup_{os.getpid()}")
            if final.exists():
                if not overwrite:
                    raise FileExistsError(final)
                if backup.exists():
                    raise FileExistsError(f"Stale backup blocks publishing: {backup}")
                os.replace(final, backup)
                backups[final] = backup
            os.replace(staged, final)
            published.append(final)
        for backup in backups.values():
            remove_path(backup)
    except Exception:
        for final in reversed(published):
            remove_path(final)
            backup = backups.get(final)
            if backup is not None and backup.exists():
                os.replace(backup, final)
        for final, backup in backups.items():
            if not final.exists() and backup.exists():
                os.replace(backup, final)
        raise


def split_selected_datasets(
    args: argparse.Namespace,
    selected: Sequence[str],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    classification_seeds = parse_seed_spec(args.classification_seeds)
    davis_kiba_seeds = parse_seed_spec(args.davis_kiba_seeds)
    bindingdb_seeds = parse_seed_spec(args.bindingdb_seeds)
    cdhit = str(preflight.get("cd-hit", ""))
    cdhit2d = str(preflight.get("cd-hit-2d", ""))
    staged_to_final: dict[Path, Path] = {}
    reports: dict[str, Any] = {}
    started = time.time()
    try:
        for name in selected:
            paths = build_dataset_paths(args.data_root, name)
            staged = stage_path(paths.output_dir)
            if staged.exists():
                raise FileExistsError(f"Stale staging directory exists: {staged}")
            staged_to_final[staged] = paths.output_dir
            if name in CLASSIFICATION_DATASETS:
                reports[name] = classification_split(
                    name, paths.inputs[0], staged, classification_seeds
                )
                continue

            if name in {"Davis", "KIBA"}:
                ligands, proteins, interactions = load_dta_csv_data(paths)
                reports[name] = create_dta_splits(
                    name,
                    ligands,
                    proteins,
                    interactions,
                    staged,
                    davis_kiba_seeds,
                    DAVIS_KIBA_RATIOS,
                    cdhit,
                    args.cdhit_threads,
                )
                continue

            ligands, proteins, interactions, binding_report = load_bindingdb_data(paths.inputs[0])
            with tempfile.TemporaryDirectory(prefix="nlrp3_") as work_dir:
                proteins, excluded = exclude_nlrp3_similar_proteins(
                    proteins, args.nlrp3_fasta, Path(work_dir), cdhit2d, args.cdhit_threads
                )
            retained_target_ids = set(proteins["target_id"].astype(str))
            before_filter = len(interactions)
            interactions = interactions[interactions["target_id"].isin(retained_target_ids)].copy()
            retained_drug_ids = set(interactions["drug_id"].astype(str))
            ligands = ligands[ligands["drug_id"].isin(retained_drug_ids)].copy()
            binding_report["nlrp3_exclusion"] = {
                "reference": str(args.nlrp3_fasta),
                "method": "cd-hit-2d global sequence identity",
                "identity_threshold": CDHIT_IDENTITY,
                "excluded_proteins": int(len(excluded)),
                "excluded_interactions": int(before_filter - len(interactions)),
                "retained_interactions": int(len(interactions)),
                "note": "identity >= 0.40 to canonical NLRP3 is excluded before splitting",
            }
            reports[name] = create_dta_splits(
                name,
                ligands,
                proteins,
                interactions,
                staged,
                bindingdb_seeds,
                BINDINGDB_RATIOS,
                cdhit,
                args.cdhit_threads,
                binding_report,
            )

        publish_outputs(staged_to_final, args.overwrite)
    except Exception:
        for staged in staged_to_final:
            if staged.exists():
                remove_path(staged)
        raise

    combined = {
        "pipeline": "five_dataset_splitting",
        "selected_datasets": list(selected),
        "elapsed_seconds": round(time.time() - started, 3),
        "datasets": reports,
    }
    return combined


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split preprocessed B3DB, BBBP, Davis, KIBA, and BindingDB datasets."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help="Datasets to split; default: all five.",
    )
    parser.add_argument("--nlrp3-fasta", type=Path, default=DEFAULT_NLRP3_FASTA)
    parser.add_argument("--classification-seeds", default="1-5")
    parser.add_argument("--davis-kiba-seeds", default="1-5")
    parser.add_argument("--bindingdb-seeds", default="1-5")
    parser.add_argument("--cdhit-threads", type=int, default=CDHIT_THREADS)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--show-rdkit-warnings", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    try:
        selected = normalize_dataset_names(args.datasets)
        if args.cdhit_threads < 0:
            raise ValueError("--cdhit-threads must be >= 0; use 0 for all CPUs")
        validate_ratios(CLASSIFICATION_RATIOS, "B3DB/BBBP")
        validate_ratios(DAVIS_KIBA_RATIOS, "Davis/KIBA")
        validate_ratios(BINDINGDB_RATIOS, "BindingDB")
        parse_seed_spec(args.classification_seeds)
        parse_seed_spec(args.davis_kiba_seeds)
        parse_seed_spec(args.bindingdb_seeds)
        if RDLogger is not None and not args.show_rdkit_warnings:
            RDLogger.DisableLog("rdApp.*")

        print(f"[Preflight] Validating preprocessed inputs for: {', '.join(selected)}")
        preflight = run_preflight(
            selected,
            args.data_root,
            args.nlrp3_fasta,
            args.overwrite,
            check_outputs=not args.check_only,
        )
        print("Preflight passed.")
        if args.check_only:
            print(json.dumps(preflight, ensure_ascii=False, indent=2))
            return 0

        split_selected_datasets(args, selected, preflight)
        print("All selected datasets were split successfully.")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
