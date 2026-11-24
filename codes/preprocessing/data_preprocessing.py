#!/usr/bin/env python3
"""Validate and preprocess the B3DB, BBBP, Davis, KIBA, and BindingDB datasets.

All required raw files and basic schemas are validated before output is
written. Processed datasets and a combined report are stored under
``./data/datasets`` by default. Pass ``--datasets`` to preprocess only a subset
of the five datasets (B3DB, BBBP, Davis, KIBA, BindingDB); preflight and
output checks follow the same selection. Dataset splitting and feature
generation are handled by separate preprocessing scripts. The BindingDB
output additionally contains a compound-protein pair table (联合去重: mean pKd
per ligand-protein pair) following the project's established cleaning protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import sqlite3
import sys
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

try:
    import numpy as np
    import pandas as pd
except ImportError:  # pragma: no cover - depends on the runtime environment
    np = None  # type: ignore[assignment]
    pd = None  # type: ignore[assignment]

try:
    from rdkit import Chem, RDLogger
except ImportError:  # pragma: no cover - depends on the runtime environment
    Chem = None  # type: ignore[assignment]
    RDLogger = None  # type: ignore[assignment]

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - progress bars are optional
    def tqdm(iterable: Iterable[Any], **_: Any) -> Iterable[Any]:
        return iterable


DEFAULT_DATA_ROOT = Path("./data/datasets")

ALL_DATASETS = ("B3DB", "BBBP", "Davis", "KIBA", "BindingDB")

SMILES_CANDIDATES = (
    "SMILES", "smiles", "Smiles", "smile", "canonical_smiles", "mol", "Drug"
)
LABEL_CANDIDATES = (
    "label", "Label", "p_np", "y", "Y", "target", "Target", "BBB+/BBB-"
)

BINDING_TABLE = "binding_data"
BINDING_SMILES = "Ligand SMILES"
BINDING_MONOMER_ID = "BindingDB MonomerID"
BINDING_SEQUENCE = "BindingDB Target Chain Sequence"
BINDING_SEQUENCE_HASH = "Sequence Hash"
BINDING_KD = "Kd (nM)"
BINDING_ORGANISM = "Target Source Organism According to Curator or DataSource"
BINDING_REQUIRED_PROPERTIES = frozenset(
    {BINDING_MONOMER_ID, BINDING_SEQUENCE, BINDING_KD, BINDING_ORGANISM}
)
HUMAN_ORGANISMS = frozenset({"homo sapiens", "human"})


@dataclass(frozen=True)
class ProjectLayout:
    data_root: Path
    b3db_raw: Path
    bbbp_raw: Path
    davis_dir: Path
    kiba_dir: Path
    bindingdb_raw: Path
    b3db_output: Path
    bbbp_output: Path
    davis_outputs: dict[str, Path]
    kiba_outputs: dict[str, Path]
    bindingdb_output: Path
    report_output: Path

    def output_paths(self) -> list[Path]:
        return [
            self.b3db_output,
            self.bbbp_output,
            *self.davis_outputs.values(),
            *self.kiba_outputs.values(),
            self.bindingdb_output,
            self.report_output,
        ]


def build_layout(data_root: Path) -> ProjectLayout:
    davis_dir = data_root / "Davis"
    kiba_dir = data_root / "KIBA"
    return ProjectLayout(
        data_root=data_root,
        b3db_raw=data_root / "B3DB" / "B3DB_classification.tsv",
        bbbp_raw=data_root / "BBBP" / "BBBP.csv",
        davis_dir=davis_dir,
        kiba_dir=kiba_dir,
        bindingdb_raw=data_root / "BindingDB" / "BindingDB_All_2D.sdf",
        b3db_output=data_root / "B3DB" / "b3db_preprocessed.csv",
        bbbp_output=data_root / "BBBP" / "bbbp_preprocessed.csv",
        davis_outputs={
            "ligands": davis_dir / "ligands_preprocessed.csv",
            "proteins": davis_dir / "proteins_preprocessed.csv",
            "interactions": davis_dir / "interactions_preprocessed.csv",
        },
        kiba_outputs={
            "ligands": kiba_dir / "ligands_preprocessed.csv",
            "proteins": kiba_dir / "proteins_preprocessed.csv",
            "interactions": kiba_dir / "interactions_preprocessed.csv",
        },
        bindingdb_output=data_root / "BindingDB" / "bindingdb_preprocessed.db",
        report_output=data_root / "preprocessing_report.json",
    )


def require_dependencies() -> None:
    missing = []
    if np is None:
        missing.append("numpy")
    if pd is None:
        missing.append("pandas")
    if Chem is None:
        missing.append("rdkit")
    if missing:
        raise RuntimeError(
            "Missing preprocessing dependencies: " + ", ".join(missing) + ". "
            "Install them before running this command."
        )


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none"}:
        return None
    return text


def find_column(columns: Sequence[Any], candidates: Sequence[str], role: str) -> str:
    lower_to_real = {str(column).casefold(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.casefold() in lower_to_real:
            return lower_to_real[candidate.casefold()]
    raise ValueError(
        f"Could not identify the {role} column. Columns={list(columns)}, "
        f"candidates={list(candidates)}"
    )


def canonicalize_smiles(value: Any, require_multiple_atoms: bool = True) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    try:
        mol = Chem.MolFromSmiles(text)
        if mol is None or (require_multiple_atoms and mol.GetNumAtoms() <= 1):
            return None
        return clean_text(Chem.MolToSmiles(mol, canonical=True))
    except Exception:
        return None


def normalize_label(value: Any) -> int | None:
    if value is None or (pd is not None and pd.isna(value)):
        return None
    if isinstance(value, str):
        normalized = value.strip().casefold()
        positive = {"1", "1.0", "true", "yes", "bbb+", "positive", "pos", "p"}
        negative = {"0", "0.0", "false", "no", "bbb-", "negative", "neg", "n"}
        if normalized in positive:
            return 1
        if normalized in negative:
            return 0
        value = normalized
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return int(numeric) if numeric in {0.0, 1.0} else None


def normalize_sequence(value: Any) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    sequence = "".join(text.split()).upper()
    return sequence or None


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def resolve_ligand_json(dataset_dir: Path) -> Path | None:
    for name in ("ligands_can.txt", "ligands_iso.txt"):
        candidate = dataset_dir / name
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def require_ligand_json(dataset_dir: Path) -> Path:
    ligand_path = resolve_ligand_json(dataset_dir)
    if ligand_path is None:
        raise FileNotFoundError(
            f"No ligand file found in {dataset_dir}. Expected ligands_can.txt "
            "or ligands_iso.txt."
        )
    return ligand_path


def load_ordered_json(path: Path) -> OrderedDict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=OrderedDict)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Expected a non-empty JSON object: {path}")
    return value


def load_y_matrix(path: Path) -> Any:
    with path.open("rb") as stream:
        try:
            value = pickle.load(stream, encoding="latin1")
        except TypeError:
            stream.seek(0)
            value = pickle.load(stream)
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"Affinity matrix must be 2D: {path}, shape={matrix.shape}")
    return matrix


def sdf_records(path: Path) -> Iterator[Any | None]:
    with path.open("rb") as stream:
        supplier = Chem.ForwardSDMolSupplier(
            stream,
            sanitize=False,
            removeHs=False,
            strictParsing=False,
        )
        yield from supplier


def check_required_files(layout: ProjectLayout, datasets: Sequence[str]) -> dict[str, Path]:
    required: dict[str, Path] = {}
    problems = []

    def _require(label: str, path: Path) -> None:
        if not path.is_file():
            problems.append(f"missing: {label}: {path}")
        elif path.stat().st_size <= 0:
            problems.append(f"empty: {label}: {path}")

    if "B3DB" in datasets:
        _require("B3DB raw TSV", layout.b3db_raw)
    if "BBBP" in datasets:
        _require("BBBP raw CSV", layout.bbbp_raw)
    if "Davis" in datasets:
        _require("Davis proteins", layout.davis_dir / "proteins.txt")
        _require("Davis affinity matrix", layout.davis_dir / "Y")
        davis_ligands = resolve_ligand_json(layout.davis_dir)
        if davis_ligands is None:
            problems.append(
                f"missing: Davis ligand JSON: {layout.davis_dir}/ligands_can.txt or ligands_iso.txt"
            )
        else:
            required["Davis ligands"] = davis_ligands
    if "KIBA" in datasets:
        _require("KIBA proteins", layout.kiba_dir / "proteins.txt")
        _require("KIBA affinity matrix", layout.kiba_dir / "Y")
        kiba_ligands = resolve_ligand_json(layout.kiba_dir)
        if kiba_ligands is None:
            problems.append(
                f"missing: KIBA ligand JSON: {layout.kiba_dir}/ligands_can.txt or ligands_iso.txt"
            )
        else:
            required["KIBA ligands"] = kiba_ligands
    if "BindingDB" in datasets:
        _require("BindingDB raw SDF", layout.bindingdb_raw)

    if problems:
        raise FileNotFoundError("Raw dataset preflight failed:\n- " + "\n- ".join(problems))
    return required


def validate_classification_schema(path: Path, separator: str) -> dict[str, Any]:
    header = pd.read_csv(path, sep=separator, nrows=0)
    smiles_col = find_column(header.columns, SMILES_CANDIDATES, "SMILES")
    label_col = find_column(header.columns, LABEL_CANDIDATES, "label")
    return {"path": str(path), "smiles_column": smiles_col, "label_column": label_col}


def validate_dta_schema(dataset_dir: Path, ligand_path: Path) -> dict[str, Any]:
    ligands = load_ordered_json(ligand_path)
    proteins = load_ordered_json(dataset_dir / "proteins.txt")
    y = load_y_matrix(dataset_dir / "Y")
    expected = (len(ligands), len(proteins))
    if tuple(y.shape) != expected:
        raise ValueError(
            f"DTA matrix shape mismatch in {dataset_dir}: Y={y.shape}, expected={expected}"
        )
    return {
        "ligand_file": str(ligand_path),
        "ligands": len(ligands),
        "proteins": len(proteins),
        "y_shape": list(y.shape),
        "finite_affinities": int(np.isfinite(y).sum()),
    }


def validate_bindingdb_schema(path: Path, scan_limit: int = 100) -> dict[str, Any]:
    seen_properties: set[str] = set()
    readable = 0
    for mol in sdf_records(path):
        if mol is None:
            continue
        readable += 1
        seen_properties.update(str(name) for name in mol.GetPropNames())
        if BINDING_REQUIRED_PROPERTIES <= seen_properties or readable >= scan_limit:
            break
    if readable == 0:
        raise ValueError(f"No readable molecule found in BindingDB SDF: {path}")
    missing = sorted(BINDING_REQUIRED_PROPERTIES - seen_properties)
    if missing:
        raise ValueError(f"BindingDB SDF is missing required properties: {missing}")
    return {"path": str(path), "readable_records_sampled": readable}


def run_preflight(layout: ProjectLayout, datasets: Sequence[str]) -> dict[str, Any]:
    print(f"[Preflight 1/2] Checking required raw files for: {', '.join(datasets)}...")
    resolved = check_required_files(layout, datasets)
    require_dependencies()

    print("[Preflight 2/2] Checking headers, JSON objects, matrices, and SDF schema...")
    report: dict[str, Any] = {}
    if "B3DB" in datasets:
        report["B3DB"] = validate_classification_schema(layout.b3db_raw, "\t")
    if "BBBP" in datasets:
        report["BBBP"] = validate_classification_schema(layout.bbbp_raw, ",")
    if "Davis" in datasets:
        report["Davis"] = validate_dta_schema(layout.davis_dir, resolved["Davis ligands"])
    if "KIBA" in datasets:
        report["KIBA"] = validate_dta_schema(layout.kiba_dir, resolved["KIBA ligands"])
    if "BindingDB" in datasets:
        report["BindingDB"] = validate_bindingdb_schema(layout.bindingdb_raw)
    print(f"Preflight passed for: {', '.join(datasets)}")
    return report


def temp_path(final_path: Path) -> Path:
    return final_path.with_name(final_path.name + ".tmp")


def clean_classification_dataset(
    input_path: Path,
    output_path: Path,
    separator: str,
    dataset_name: str,
) -> dict[str, Any]:
    frame = pd.read_csv(input_path, sep=separator)
    smiles_col = find_column(frame.columns, SMILES_CANDIDATES, "SMILES")
    label_col = find_column(frame.columns, LABEL_CANDIDATES, "label")
    raw_rows = len(frame)

    work = frame[[smiles_col, label_col]].copy()
    work.columns = ["SMILES", "raw_label"]
    missing_input = int(work[["SMILES", "raw_label"]].isna().any(axis=1).sum())
    work = work.dropna(subset=["SMILES", "raw_label"]).copy()

    work["label"] = work["raw_label"].map(normalize_label)
    invalid_label = int(work["label"].isna().sum())
    work = work.dropna(subset=["label"]).copy()
    work["label"] = work["label"].astype(int)

    work["SMILES"] = work["SMILES"].map(canonicalize_smiles)
    invalid_smiles = int(work["SMILES"].isna().sum())
    work = work.dropna(subset=["SMILES"]).copy()

    conflicting_smiles = int(
        (work.groupby("SMILES")["label"].nunique() > 1).sum()
    )
    before_dedup = len(work)
    work = work.drop_duplicates(subset=["SMILES"], keep="first")
    duplicates_removed = before_dedup - len(work)
    work = work[["SMILES", "label"]].reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    work.to_csv(output_path, index=False)
    return {
        "dataset": dataset_name,
        "input": str(input_path),
        "output": str(output_path),
        "raw_rows": raw_rows,
        "missing_smiles_or_label": missing_input,
        "invalid_labels": invalid_label,
        "invalid_or_single_atom_smiles": invalid_smiles,
        "conflicting_duplicate_smiles": conflicting_smiles,
        "duplicate_rows_removed": duplicates_removed,
        "retained_rows": len(work),
        "label_distribution": {
            str(key): int(value) for key, value in work["label"].value_counts().sort_index().items()
        },
    }


def prepare_davis_affinity(y_raw: Any) -> tuple[Any, dict[str, Any]]:
    finite = y_raw[np.isfinite(y_raw)]
    if finite.size == 0:
        raise ValueError("Davis Y contains no finite affinity values")
    median = float(np.median(finite))
    p95 = float(np.percentile(finite, 95))
    looks_like_kd_nm = p95 > 50.0 or median > 20.0
    if not looks_like_kd_nm:
        return y_raw.astype(float).copy(), {
            "affinity_label": "pKd",
            "conversion": "none; input already resembles pKd",
        }

    converted = np.full_like(y_raw, np.nan, dtype=float)
    valid = np.isfinite(y_raw) & (y_raw > 0)
    converted[valid] = 9.0 - np.log10(y_raw[valid])
    return converted, {
        "affinity_label": "pKd",
        "conversion": "pKd = 9 - log10(Kd_nM)",
    }


def clean_dta_dataset(
    dataset_dir: Path,
    ligand_path: Path,
    staged_outputs: dict[str, Path],
    dataset_name: str,
) -> dict[str, Any]:
    ligands = load_ordered_json(ligand_path)
    proteins = load_ordered_json(dataset_dir / "proteins.txt")
    y_raw = load_y_matrix(dataset_dir / "Y")
    if y_raw.shape != (len(ligands), len(proteins)):
        raise ValueError(f"{dataset_name} Y shape changed after preflight")

    drug_ids = [str(key) for key in ligands]
    target_ids = [str(key) for key in proteins]
    cleaned_smiles = [canonicalize_smiles(value) for value in ligands.values()]
    cleaned_sequences = [normalize_sequence(value) for value in proteins.values()]
    valid_drug = np.asarray([value is not None for value in cleaned_smiles], dtype=bool)
    valid_target = np.asarray([value is not None for value in cleaned_sequences], dtype=bool)

    if dataset_name == "Davis":
        y, affinity_info = prepare_davis_affinity(y_raw)
    else:
        y = y_raw.astype(float).copy()
        affinity_info = {"affinity_label": "KIBA score", "conversion": "none"}

    valid_pairs = np.isfinite(y) & valid_drug[:, None] & valid_target[None, :]
    drug_indices, target_indices = np.where(valid_pairs)
    if len(drug_indices) == 0:
        raise RuntimeError(f"No valid interactions remain after cleaning {dataset_name}")

    used_drugs = set(int(index) for index in np.unique(drug_indices))
    used_targets = set(int(index) for index in np.unique(target_indices))
    ligand_frame = pd.DataFrame(
        [
            {"drug_id": drug_ids[index], "SMILES": cleaned_smiles[index]}
            for index in range(len(drug_ids)) if index in used_drugs
        ]
    )
    protein_frame = pd.DataFrame(
        [
            {
                "target_id": target_ids[index],
                "Protein Sequence": cleaned_sequences[index],
                "Sequence Hash": sequence_hash(cleaned_sequences[index]),
            }
            for index in range(len(target_ids)) if index in used_targets
        ]
    )
    interaction_frame = pd.DataFrame(
        {
            "drug_id": [drug_ids[int(index)] for index in drug_indices],
            "target_id": [target_ids[int(index)] for index in target_indices],
            "affinity": y[drug_indices, target_indices].astype(float),
        }
    )

    for path in staged_outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    ligand_frame.to_csv(staged_outputs["ligands"], index=False)
    protein_frame.to_csv(staged_outputs["proteins"], index=False)
    interaction_frame.to_csv(staged_outputs["interactions"], index=False)

    return {
        "dataset": dataset_name,
        "raw_ligands": len(ligands),
        "raw_proteins": len(proteins),
        "raw_finite_interactions": int(np.isfinite(y_raw).sum()),
        "invalid_or_single_atom_ligands": int((~valid_drug).sum()),
        "empty_proteins": int((~valid_target).sum()),
        "retained_ligands": len(ligand_frame),
        "retained_proteins": len(protein_frame),
        "retained_interactions": len(interaction_frame),
        **affinity_info,
    }


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def get_sdf_property(mol: Any, name: str) -> str | None:
    try:
        return clean_text(mol.GetProp(name)) if mol.HasProp(name) else None
    except Exception:
        return None


def normalize_monomer_id(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except ValueError:
        return None
    if not math.isfinite(numeric) or numeric <= 0 or not numeric.is_integer():
        return None
    return str(int(numeric))


def parse_exact_kd(value: str | None) -> float | None:
    if value is None or "<" in value or ">" in value:
        return None
    try:
        kd = float(value)
    except ValueError:
        return None
    return kd if math.isfinite(kd) and kd > 0 else None


def bindingdb_smiles(mol: Any) -> str | None:
    property_value = get_sdf_property(mol, BINDING_SMILES)
    parsed = None
    if property_value is not None:
        try:
            parsed = Chem.MolFromSmiles(property_value)
        except Exception:
            parsed = None
    if parsed is None:
        try:
            fallback = Chem.MolToSmiles(mol, canonical=True)
            parsed = Chem.MolFromSmiles(fallback)
        except Exception:
            return None
    if parsed is None or parsed.GetNumAtoms() <= 1:
        return None
    return clean_text(Chem.MolToSmiles(parsed, canonical=True))


def extract_bindingdb_row(
    mol: Any,
) -> tuple[tuple[str, str, str, str, float] | None, str | None]:
    organism = get_sdf_property(mol, BINDING_ORGANISM)
    if organism is None or organism.casefold() not in HUMAN_ORGANISMS:
        return None, "non_human_target"

    kd_raw = get_sdf_property(mol, BINDING_KD)
    if kd_raw is None:
        return None, "missing_kd"
    if "<" in kd_raw or ">" in kd_raw:
        return None, "censored_kd"
    kd = parse_exact_kd(kd_raw)
    if kd is None:
        return None, "invalid_kd"

    monomer_id = normalize_monomer_id(get_sdf_property(mol, BINDING_MONOMER_ID))
    if monomer_id is None:
        return None, "missing_or_invalid_monomer_id"
    sequence = normalize_sequence(get_sdf_property(mol, BINDING_SEQUENCE))
    if sequence is None:
        return None, "missing_sequence"
    smiles = bindingdb_smiles(mol)
    if smiles is None:
        return None, "missing_or_invalid_smiles"
    return (monomer_id, smiles, sequence, sequence_hash(sequence), kd), None


def create_bindingdb_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        f"""
        CREATE TABLE {quote_identifier(BINDING_TABLE)} (
            "id" INTEGER PRIMARY KEY AUTOINCREMENT,
            {quote_identifier(BINDING_MONOMER_ID)} TEXT NOT NULL,
            {quote_identifier(BINDING_SMILES)} TEXT NOT NULL,
            {quote_identifier(BINDING_SEQUENCE)} TEXT NOT NULL,
            {quote_identifier(BINDING_SEQUENCE_HASH)} TEXT NOT NULL,
            {quote_identifier(BINDING_KD)} REAL NOT NULL CHECK ({quote_identifier(BINDING_KD)} > 0)
        );
        CREATE INDEX "idx_binding_monomer_id"
            ON {quote_identifier(BINDING_TABLE)} ({quote_identifier(BINDING_MONOMER_ID)});
        CREATE INDEX "idx_binding_sequence_hash"
            ON {quote_identifier(BINDING_TABLE)} ({quote_identifier(BINDING_SEQUENCE_HASH)});
        """
    )


def build_bindingdb_lookup_tables(conn: sqlite3.Connection) -> None:
    q_table = quote_identifier(BINDING_TABLE)
    q_mid = quote_identifier(BINDING_MONOMER_ID)
    q_smiles = quote_identifier(BINDING_SMILES)
    q_hash = quote_identifier(BINDING_SEQUENCE_HASH)
    q_sequence = quote_identifier(BINDING_SEQUENCE)
    conn.executescript(
        f"""
        CREATE TABLE "ligand" AS
        SELECT {q_mid}, MIN({q_smiles}) AS {q_smiles}
        FROM {q_table} GROUP BY {q_mid};
        CREATE UNIQUE INDEX "idx_ligand_monomer_id" ON "ligand" ({q_mid});

        CREATE TABLE "protein" AS
        SELECT {q_hash}, MIN({q_sequence}) AS {q_sequence}
        FROM {q_table} GROUP BY {q_hash};
        CREATE UNIQUE INDEX "idx_protein_sequence_hash" ON "protein" ({q_hash});
        """
    )


def build_bindingdb_pair_table(conn: sqlite3.Connection) -> None:
    """化合物-蛋白联合去重（pair 级）：按 (BindingDB MonomerID, Sequence Hash) 分组。

    参照项目既有 BindingDB 清洗管线（X.12）的口径：
      pKd = 9 - log10(Kd[nM])，行级保留 0 < pKd < 15，
      按 pair 分组后取 pKd 均值与测量次数，写入 compound_protein_pair 表。
    """
    conn.executescript(
        """
        CREATE TABLE "compound_protein_pair" AS
        SELECT
            "BindingDB MonomerID"          AS "MonomerID",
            "Sequence Hash"                AS "SequenceHash",
            AVG(9.0 - log10("Kd (nM)"))    AS "pKd_mean",
            COUNT(*)                       AS "measurement_count"
        FROM "binding_data"
        WHERE "Kd (nM)" > 0
          AND (9.0 - log10("Kd (nM)")) > 0
          AND (9.0 - log10("Kd (nM)")) < 15
        GROUP BY "BindingDB MonomerID", "Sequence Hash";
        CREATE UNIQUE INDEX "idx_compound_protein_pair"
            ON "compound_protein_pair" ("MonomerID", "SequenceHash");
        """
    )


def scalar(conn: sqlite3.Connection, sql: str) -> int:
    row = conn.execute(sql).fetchone()
    if row is None:
        raise RuntimeError(f"Query returned no result: {sql}")
    return int(row[0])


def clean_bindingdb(input_path: Path, output_path: Path, batch_size: int) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(output_path)
    columns = [
        BINDING_MONOMER_ID,
        BINDING_SMILES,
        BINDING_SEQUENCE,
        BINDING_SEQUENCE_HASH,
        BINDING_KD,
    ]
    column_sql = ", ".join(quote_identifier(column) for column in columns)
    insert_sql = (
        f"INSERT INTO {quote_identifier(BINDING_TABLE)} ({column_sql}) "
        "VALUES (?, ?, ?, ?, ?)"
    )
    counts: Counter[str] = Counter()
    batch: list[tuple[str, str, str, str, float]] = []
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        create_bindingdb_schema(conn)
        for mol in tqdm(sdf_records(input_path), desc="Preprocessing BindingDB", unit="mol"):
            counts["source_records"] += 1
            if mol is None:
                counts["unreadable_sdf_record"] += 1
                continue
            row, rejection = extract_bindingdb_row(mol)
            if rejection is not None:
                counts[rejection] += 1
                continue
            batch.append(row)
            if len(batch) >= batch_size:
                conn.executemany(insert_sql, batch)
                conn.commit()
                counts["retained"] += len(batch)
                batch.clear()
        if batch:
            conn.executemany(insert_sql, batch)
            conn.commit()
            counts["retained"] += len(batch)

        build_bindingdb_lookup_tables(conn)
        build_bindingdb_pair_table(conn)
        conn.commit()
        conn.execute("ANALYZE")
        conn.commit()
        integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity.casefold() != "ok":
            raise RuntimeError(f"BindingDB SQLite integrity check failed: {integrity}")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("VACUUM")
        conn.commit()
        database_stats = {
            "integrity_check": integrity,
            "binding_rows": scalar(conn, 'SELECT COUNT(*) FROM "binding_data"'),
            "ligands": scalar(conn, 'SELECT COUNT(*) FROM "ligand"'),
            "proteins": scalar(conn, 'SELECT COUNT(*) FROM "protein"'),
            "compound_protein_pairs": scalar(
                conn, 'SELECT COUNT(*) FROM "compound_protein_pair"'
            ),
        }
    finally:
        conn.close()
    return {
        "dataset": "BindingDB",
        "input": str(input_path),
        "output": str(output_path),
        "row_filter_counts": dict(sorted(counts.items())),
        "database": database_stats,
    }


def layout_outputs(layout: ProjectLayout, datasets: Sequence[str]) -> list[Path]:
    outputs = [layout.report_output]
    if "B3DB" in datasets:
        outputs.append(layout.b3db_output)
    if "BBBP" in datasets:
        outputs.append(layout.bbbp_output)
    if "Davis" in datasets:
        outputs.extend(layout.davis_outputs.values())
    if "KIBA" in datasets:
        outputs.extend(layout.kiba_outputs.values())
    if "BindingDB" in datasets:
        outputs.append(layout.bindingdb_output)
    return outputs


def staged_output_map(layout: ProjectLayout, datasets: Sequence[str]) -> dict[Path, Path]:
    return {temp_path(final): final for final in layout_outputs(layout, datasets)}


def remove_staged_files(staged_to_final: dict[Path, Path]) -> None:
    for staged in staged_to_final:
        for candidate in (staged, Path(str(staged) + "-wal"), Path(str(staged) + "-shm")):
            if candidate.is_file():
                candidate.unlink()


def ensure_outputs_available(layout: ProjectLayout, datasets: Sequence[str], overwrite: bool) -> None:
    existing = [path for path in layout_outputs(layout, datasets) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Preprocessed output already exists. Pass --overwrite to replace it:\n- "
            + "\n- ".join(str(path) for path in existing)
        )


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def preprocess_all(layout: ProjectLayout, args: argparse.Namespace, preflight: dict[str, Any]) -> dict[str, Any]:
    datasets = args.datasets
    ensure_outputs_available(layout, datasets, args.overwrite)
    staged_to_final = staged_output_map(layout, datasets)
    remove_staged_files(staged_to_final)
    started = time.time()

    b3db_stage = temp_path(layout.b3db_output)
    bbbp_stage = temp_path(layout.bbbp_output)
    davis_stages = {name: temp_path(path) for name, path in layout.davis_outputs.items()}
    kiba_stages = {name: temp_path(path) for name, path in layout.kiba_outputs.items()}
    bindingdb_stage = temp_path(layout.bindingdb_output)
    report_stage = temp_path(layout.report_output)

    try:
        results: dict[str, Any] = {}
        if "B3DB" in datasets:
            results["B3DB"] = clean_classification_dataset(
                layout.b3db_raw, b3db_stage, "\t", "B3DB"
            )
        if "BBBP" in datasets:
            results["BBBP"] = clean_classification_dataset(
                layout.bbbp_raw, bbbp_stage, ",", "BBBP"
            )
        if "Davis" in datasets:
            results["Davis"] = clean_dta_dataset(
                layout.davis_dir,
                require_ligand_json(layout.davis_dir),
                davis_stages,
                "Davis",
            )
        if "KIBA" in datasets:
            results["KIBA"] = clean_dta_dataset(
                layout.kiba_dir,
                require_ligand_json(layout.kiba_dir),
                kiba_stages,
                "KIBA",
            )
        if "BindingDB" in datasets:
            results["BindingDB"] = clean_bindingdb(
                layout.bindingdb_raw, bindingdb_stage, args.bindingdb_batch_size
            )

        # Record published paths rather than temporary staging paths.
        if "B3DB" in datasets:
            results["B3DB"]["output"] = str(layout.b3db_output)
        if "BBBP" in datasets:
            results["BBBP"]["output"] = str(layout.bbbp_output)
        if "Davis" in datasets:
            results["Davis"]["outputs"] = {
                name: str(path) for name, path in layout.davis_outputs.items()
            }
        if "KIBA" in datasets:
            results["KIBA"]["outputs"] = {
                name: str(path) for name, path in layout.kiba_outputs.items()
            }
        if "BindingDB" in datasets:
            results["BindingDB"]["output"] = str(layout.bindingdb_output)
        report = {
            "pipeline": "five_dataset_preprocessing",
            "data_root": str(layout.data_root),
            "datasets": list(datasets),
            "preflight": preflight,
            "results": results,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        write_json(report_stage, report)

        # Publish only after every dataset and the combined report succeeded.
        for staged, final in staged_to_final.items():
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, final)
    except Exception:
        remove_staged_files(staged_to_final)
        raise

    print(f"Preprocessed dataset(s) successfully: {', '.join(datasets)}")
    print(f"Combined report: {layout.report_output}")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight and preprocess B3DB, BBBP, Davis, KIBA, and BindingDB."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Dataset root containing the five dataset folders.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=ALL_DATASETS,
        default=list(ALL_DATASETS),
        help=(
            "Datasets to preprocess: B3DB, BBBP, Davis, KIBA, BindingDB. "
            "Default: all five."
        ),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the selected raw datasets without writing outputs.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--bindingdb-batch-size", type=int, default=10_000)
    parser.add_argument("--show-rdkit-warnings", action="store_true")
    args = parser.parse_args(argv)
    if args.bindingdb_batch_size <= 0:
        parser.error("--bindingdb-batch-size must be positive")
    if not args.show_rdkit_warnings and RDLogger is not None:
        RDLogger.DisableLog("rdApp.*")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    layout = build_layout(args.data_root)
    try:
        preflight = run_preflight(layout, args.datasets)
        if args.check_only:
            print(json.dumps(preflight, ensure_ascii=False, indent=2))
            return 0
        preprocess_all(layout, args, preflight)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, sqlite3.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
