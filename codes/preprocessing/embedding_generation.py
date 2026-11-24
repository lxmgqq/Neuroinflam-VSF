#!/usr/bin/env python3
"""Generate cached BP-NET and UE-AlignNet features for all five datasets.

With no arguments, the script processes B3DB, BBBP, Davis, KIBA, and BindingDB.
Artifacts are stored per entity under ``./data/embedding``. Existing artifacts
are reused only after their metadata, dimensions, and values pass validation;
atomic writes and final manifest reconstruction support resumable execution.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import os
import time
import uuid
from copy import copy
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


DATASETS = ("B3DB", "BBBP", "Davis", "KIBA", "BindingDB")
CLASSIFICATION_DATASETS = frozenset({"B3DB", "BBBP"})
DTA_DATASETS = frozenset({"Davis", "KIBA", "BindingDB"})
FEATURES = ("unimol2", "rdkit", "esmc", "esm2_contact_graph")
FORMAT_VERSION = 1

DEFAULT_DATA_ROOT = Path("./data/datasets")
DEFAULT_EMBEDDING_ROOT = Path("./data/embedding")
UNIMOL_DIM = 1536
ESMC_DIM = 1152
MIXED_FINGERPRINT_DIM = 2048 + 2048 + 2048 + 167


@dataclass(frozen=True)
class Entity:
    dataset: str
    entity_type: str
    entity_id: str
    text: str

    @property
    def text_sha256(self) -> str:
        return sha256_text(self.text)

    @property
    def artifact_key(self) -> str:
        raw = f"{self.entity_type}\0{self.entity_id}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:24]


@dataclass
class ScanResult:
    rows: dict[str, dict[str, Any]]
    pending: list[Entity]
    valid_count: int


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_id(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if text.endswith(".0"):
        try:
            text = str(int(float(text)))
        except ValueError:
            pass
    return text


def clean_sequence(value: Any) -> str:
    sequence = "".join(str(value or "").split()).upper()
    allowed = set("ACDEFGHIKLMNPQRSTVWY")
    return "".join(character if character in allowed else "X" for character in sequence)


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"Empty {label}: {path}")


def load_classification_split_entities(
    dataset_dir: Path, dataset: str
) -> list[Entity]:
    """Collect the exact SMILES strings consumed by BP-NET training."""
    split_root = dataset_dir / "splits"
    entities: dict[str, Entity] = {}
    labels: dict[str, int] = {}
    found = 0
    if split_root.is_dir():
        for csv_path in sorted(split_root.rglob("*.csv")):
            parts = csv_path.relative_to(split_root).parts
            if (
                len(parts) != 3
                or not parts[0].startswith("seed_")
                or parts[1] != "scaffold"
                or parts[2] not in ("train.csv", "val.csv", "test.csv")
            ):
                continue
            found += 1
            expected_split = csv_path.stem
            with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                missing = {"SMILES", "label", "Split"} - set(
                    reader.fieldnames or ()
                )
                if missing:
                    raise ValueError(
                        f"{csv_path} is missing columns: {sorted(missing)}"
                    )
                for row_number, row in enumerate(reader, start=2):
                    smiles = str(row.get("SMILES", "") or "").strip()
                    if not smiles:
                        raise ValueError(f"Empty SMILES in {csv_path}:{row_number}")
                    try:
                        label_value = float(str(row.get("label", "")).strip())
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid label in {csv_path}:{row_number}"
                        ) from exc
                    if label_value not in (0.0, 1.0):
                        raise ValueError(
                            f"Label outside {{0,1}} in {csv_path}:{row_number}"
                        )
                    split_value = str(row.get("Split", "") or "").strip().lower()
                    if split_value != expected_split:
                        raise ValueError(
                            f"Split column mismatch in {csv_path}:{row_number}: "
                            f"expected {expected_split!r}, found {split_value!r}"
                        )
                    entity_id = sha256_text(smiles)
                    previous = entities.get(entity_id)
                    if previous is not None and previous.text != smiles:
                        raise ValueError(
                            f"Conflicting SMILES for entity ID {entity_id!r} in "
                            f"{csv_path}:{row_number}"
                        )
                    label = int(label_value)
                    previous_label = labels.get(entity_id)
                    if previous_label is not None and previous_label != label:
                        raise ValueError(
                            f"Conflicting labels for SMILES {smiles!r} in "
                            f"{csv_path}:{row_number}"
                        )
                    entities[entity_id] = Entity(
                        dataset, "ligand", entity_id, smiles
                    )
                    labels[entity_id] = label
    if not found or not entities:
        raise FileNotFoundError(
            f"No scaffold split CSVs found under {split_root}. Run "
            f"data_splitting.py or provide the precomputed splits before "
            f"embedding generation for {dataset}."
        )
    return sorted(entities.values(), key=lambda item: item.entity_id)


def _load_csv_text_map(
    path: Path, id_column: str, text_column: str, label: str
) -> dict[str, str]:
    """Read a cleaned Davis/KIBA entity table."""
    require_file(path, label)
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        missing = {id_column, text_column} - columns
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            entity_id = normalize_id(row.get(id_column))
            text = str(row.get(text_column, "") or "").strip()
            if text_column == "Protein Sequence":
                text = clean_sequence(text)
            if entity_id and text:
                result[entity_id] = text
    return result


def _load_bindingdb_text_maps(
    db_path: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    """Read ligand and protein lookup tables from the cleaned BindingDB database."""
    require_file(db_path, "BindingDB cleaned database")
    import sqlite3

    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        ligand_rows = conn.execute(
            'SELECT "BindingDB MonomerID", MIN("Ligand SMILES") '
            'FROM "ligand" GROUP BY "BindingDB MonomerID"'
        ).fetchall()
        protein_rows = conn.execute(
            'SELECT "Sequence Hash", MIN("BindingDB Target Chain Sequence") '
            'FROM "protein" GROUP BY "Sequence Hash"'
        ).fetchall()
    finally:
        conn.close()
    ligand_text: dict[str, str] = {}
    for monomer_id, smiles in ligand_rows:
        entity_id = normalize_id(monomer_id)
        text = str(smiles or "").strip()
        if entity_id and text:
            ligand_text[entity_id] = text
    protein_text: dict[str, str] = {}
    for sequence_hash, sequence in protein_rows:
        entity_id = normalize_id(sequence_hash)
        text = clean_sequence(sequence)
        if entity_id and text:
            protein_text[entity_id] = text
    return ligand_text, protein_text


def _build_entities(
    dataset: str,
    entity_type: str,
    entity_ids: set[str],
    text_map: Mapping[str, str],
) -> list[Entity]:
    entities: dict[str, Entity] = {}
    for entity_id in sorted(entity_ids):
        text = text_map.get(entity_id)
        if text is None:
            raise ValueError(
                f"Missing {entity_type} text for {entity_id!r} in {dataset}; "
                f"ensure the cleaned dataset files contain this ID"
            )
        entity = Entity(dataset, entity_type, entity_id, text)
        previous = entities.get(entity_id)
        if previous is not None and previous.text != text:
            raise ValueError(f"Conflicting {entity_type} text for ID {entity_id!r} in {dataset}")
        entities[entity_id] = entity
    return sorted(entities.values(), key=lambda item: item.entity_id)


def load_dataset_entities(data_root: Path, dataset: str) -> dict[str, list[Entity]]:
    dataset_dir = data_root / dataset
    if dataset in CLASSIFICATION_DATASETS:
        return {
            "ligand": load_classification_split_entities(dataset_dir, dataset)
        }

    # DTA datasets: collect entity IDs directly from the precomputed split CSVs
    # (train/val/test across every seed and scenario). Molecular text is then
    # resolved from the cleaned dataset files produced by data_preprocessing.py,
    # so embedding depends on nothing else (no clustering/entity/raw tables).
    scenario_names = ("warm", "cold_drug", "cold_target", "double_cold")
    ligand_ids: set[str] = set()
    protein_ids: set[str] = set()
    split_root = dataset_dir / "splits"
    found = 0
    if split_root.is_dir():
        for csv_path in sorted(split_root.rglob("*.csv")):
            parts = csv_path.relative_to(split_root).parts
            if (
                len(parts) != 3
                or not parts[0].startswith("seed_")
                or parts[1] not in scenario_names
                or parts[2] not in ("train.csv", "val.csv", "test.csv")
            ):
                continue
            found += 1
            with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                missing = {"drug_id", "target_id"} - set(reader.fieldnames or ())
                if missing:
                    raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
                for row in reader:
                    if row.get("drug_id"):
                        ligand_ids.add(normalize_id(row["drug_id"]))
                    if row.get("target_id"):
                        protein_ids.add(normalize_id(row["target_id"]))
    if not found or not ligand_ids or not protein_ids:
        raise FileNotFoundError(
            f"No split CSVs found under {split_root}. Run data_splitting.py or "
            f"provide precomputed splits before embedding generation for {dataset}."
        )

    try:
        if dataset in ("Davis", "KIBA"):
            # Text comes from the cleaned preprocessed CSVs.
            ligand_text = _load_csv_text_map(
                dataset_dir / "ligands_preprocessed.csv",
                "drug_id",
                "SMILES",
                f"{dataset} cleaned ligand table",
            )
            protein_text = _load_csv_text_map(
                dataset_dir / "proteins_preprocessed.csv",
                "target_id",
                "Protein Sequence",
                f"{dataset} cleaned protein table",
            )
        else:  # BindingDB: text comes from the cleaned SQLite database.
            ligand_text, protein_text = _load_bindingdb_text_maps(
                dataset_dir / "bindingdb_preprocessed.db"
            )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{exc}\nRun data_preprocessing.py before embedding generation for {dataset}."
        ) from exc

    ligands = _build_entities(dataset, "ligand", ligand_ids, ligand_text)
    proteins = _build_entities(dataset, "protein", protein_ids, protein_text)
    return {"ligand": ligands, "protein": proteins}


def normalize_dataset_names(values: Sequence[str]) -> list[str]:
    if not values or any(value.casefold() == "all" for value in values):
        return list(DATASETS)
    lookup = {name.casefold(): name for name in DATASETS}
    result: list[str] = []
    for value in values:
        key = value.casefold()
        if key not in lookup:
            raise ValueError(f"Unknown dataset {value!r}; choose from {DATASETS} or all")
        if lookup[key] not in result:
            result.append(lookup[key])
    return result


def normalize_feature_names(values: Sequence[str]) -> list[str]:
    aliases = {
        "unimol": "unimol2",
        "unimol2": "unimol2",
        "rdkit": "rdkit",
        "graph": "rdkit",
        "esmc": "esmc",
        "esm2": "esm2_contact_graph",
        "esm2_contact_graph": "esm2_contact_graph",
    }
    if not values or any(value.casefold() == "all" for value in values):
        return list(FEATURES)
    result: list[str] = []
    for value in values:
        key = value.casefold()
        if key not in aliases:
            raise ValueError(f"Unknown feature {value!r}; choose from {FEATURES} or all")
        if aliases[key] not in result:
            result.append(aliases[key])
    return result


def feature_entities(
    dataset: str,
    entities: Mapping[str, list[Entity]],
    feature: str,
) -> list[Entity]:
    if feature in {"unimol2", "rdkit"}:
        return list(entities.get("ligand", ()))
    if feature in {"esmc", "esm2_contact_graph"} and dataset in DTA_DATASETS:
        return list(entities.get("protein", ()))
    return []


def feature_directory(root: Path, dataset: str, feature: str) -> Path:
    entity_type = "ligand" if feature in {"unimol2", "rdkit"} else "protein"
    return root / dataset / entity_type / feature


def artifact_path(root: Path, entity: Entity, feature: str) -> Path:
    suffix = ".npz" if feature == "esm2_contact_graph" else ".pt"
    return feature_directory(root, entity.dataset, feature) / f"{entity.artifact_key}{suffix}"


def torch_load(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Compatibility fallback when weights_only is unsupported.
        return torch.load(path, map_location="cpu")


def feature_configuration(
    args: argparse.Namespace, dataset: str, feature: str
) -> dict[str, Any]:
    if feature == "unimol2":
        return {
            "model_name": args.unimol_model_name,
            "model_size": args.unimol_model_size,
            "remove_hs": False,
            "return_atomic_reprs": dataset in CLASSIFICATION_DATASETS,
        }
    if feature == "rdkit":
        return {
            "schema": (
                "bp_brics_graph+mixed_fp"
                if dataset in CLASSIFICATION_DATASETS
                else "ue_dense_graph"
            ),
            "mixed_fingerprint_dim": (
                MIXED_FINGERPRINT_DIM
                if dataset in CLASSIFICATION_DATASETS
                else None
            ),
        }
    if feature == "esmc":
        return {
            "model_name": args.esmc_model_name,
            "max_sequence_length": args.esmc_max_sequence_length,
            "embedding_dim": ESMC_DIM,
        }
    if feature == "esm2_contact_graph":
        return {
            "backend": args.esm2_backend,
            "model_name": (
                args.esm2_model_name
                if args.esm2_backend == "transformers"
                else args.esm2_fair_model_name
            ),
            "torch_dtype": (
                args.esm2_torch_dtype
                if args.esm2_backend == "transformers"
                else "model_default"
            ),
            "window_size": args.esm2_window_size,
            "window_overlap": args.esm2_window_overlap,
            "probability_threshold": args.esm2_probability_threshold,
            "top_k_long_range": args.esm2_top_k_long_range,
            "minimum_separation": args.esm2_minimum_separation,
        }
    raise ValueError(feature)


def configuration_sha256(
    args: argparse.Namespace, dataset: str, feature: str
) -> str:
    serialized = json.dumps(
        feature_configuration(args, dataset, feature),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(serialized)


def common_metadata_valid(
    obj: Any,
    entity: Entity,
    feature: str,
    expected_configuration_sha256: str,
) -> bool:
    return bool(
        isinstance(obj, dict)
        and int(obj.get("format_version", -1)) == FORMAT_VERSION
        and obj.get("dataset") == entity.dataset
        and obj.get("entity_type") == entity.entity_type
        and str(obj.get("entity_id")) == entity.entity_id
        and obj.get("feature") == feature
        and obj.get("text_sha256") == entity.text_sha256
        and obj.get("configuration_sha256") == expected_configuration_sha256
    )


def finite_tensor(value: Any) -> bool:
    import torch

    return isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all())


def validate_unimol(
    path: Path, entity: Entity, expected_configuration_sha256: str
) -> tuple[bool, str, str]:
    try:
        if not path.is_file():
            return False, "missing", ""
        obj = torch_load(path)
        if not common_metadata_valid(
            obj, entity, "unimol2", expected_configuration_sha256
        ):
            return False, "metadata_or_input_changed", ""
        cls_repr = obj.get("cls_repr")
        if not finite_tensor(cls_repr) or cls_repr.ndim != 1 or cls_repr.numel() != UNIMOL_DIM:
            return False, "invalid_cls", ""
        shape = f"cls={tuple(cls_repr.shape)}"
        if entity.dataset in CLASSIFICATION_DATASETS:
            atomic = obj.get("atomic_reprs")
            if (
                not finite_tensor(atomic)
                or atomic.ndim != 2
                or atomic.shape[0] <= 0
                or atomic.shape[1] != UNIMOL_DIM
            ):
                return False, "invalid_atomic_reprs", ""
            shape += f";atomic={tuple(atomic.shape)}"
        return True, "ok", shape
    except Exception as exc:
        return False, f"load_failed:{type(exc).__name__}", ""


def validate_rdkit(
    path: Path, entity: Entity, expected_configuration_sha256: str
) -> tuple[bool, str, str]:
    try:
        if not path.is_file():
            return False, "missing", ""
        obj = torch_load(path)
        if not common_metadata_valid(
            obj, entity, "rdkit", expected_configuration_sha256
        ):
            return False, "metadata_or_input_changed", ""
        if entity.dataset in CLASSIFICATION_DATASETS:
            required = (
                "fingerprint", "bp_x", "bp_edge_index", "bp_edge_attr",
                "atom_to_motif", "motif_edge_index", "motif_edge_attr",
                "motif_sizes",
            )
            if any(key not in obj for key in required):
                return False, "missing_fields", ""
            fp, bp_x = obj["fingerprint"], obj["bp_x"]
            edge_index, edge_attr = obj["bp_edge_index"], obj["bp_edge_attr"]
            atom_to_motif = obj["atom_to_motif"]
            motif_edge_index = obj["motif_edge_index"]
            motif_edge_attr = obj["motif_edge_attr"]
            motif_sizes = obj["motif_sizes"]
            valid = (
                finite_tensor(fp) and fp.ndim == 1
                and fp.numel() == MIXED_FINGERPRINT_DIM
                and finite_tensor(bp_x) and bp_x.ndim == 2 and bp_x.shape[0] > 1
                and finite_tensor(edge_attr) and edge_attr.ndim == 2
                and edge_index.ndim == 2 and edge_index.shape[0] == 2
                and edge_index.shape[1] == edge_attr.shape[0]
                and atom_to_motif.ndim == 1
                and atom_to_motif.shape[0] == bp_x.shape[0]
                and motif_edge_index.ndim == 2
                and motif_edge_index.shape[0] == 2
                and motif_edge_attr.ndim == 2
                and motif_edge_index.shape[1] == motif_edge_attr.shape[0]
                and motif_sizes.ndim == 2 and motif_sizes.shape[0] > 0
            )
            if not valid:
                return False, "invalid_shapes_or_values", ""
            return True, "ok", f"atoms={bp_x.shape[0]};fp={fp.numel()}"

        required = ("ue_x", "ue_adj")
        if any(key not in obj for key in required):
            return False, "missing_fields", ""
        ue_x, ue_adj = obj["ue_x"], obj["ue_adj"]
        valid = (
            finite_tensor(ue_x) and ue_x.ndim == 2 and ue_x.shape[0] > 1
            and finite_tensor(ue_adj)
            and ue_adj.shape == (ue_x.shape[0], ue_x.shape[0])
        )
        if not valid:
            return False, "invalid_shapes_or_values", ""
        return True, "ok", f"atoms={ue_x.shape[0]}"
    except Exception as exc:
        return False, f"load_failed:{type(exc).__name__}", ""


def expected_esmc_length(entity: Entity, max_sequence_length: int) -> int:
    if max_sequence_length > 0:
        return min(len(entity.text), max_sequence_length)
    return len(entity.text)


def validate_esmc(
    path: Path,
    entity: Entity,
    max_sequence_length: int,
    expected_configuration_sha256: str,
) -> tuple[bool, str, str]:
    try:
        if not path.is_file():
            return False, "missing", ""
        obj = torch_load(path)
        if not common_metadata_valid(
            obj, entity, "esmc", expected_configuration_sha256
        ):
            return False, "metadata_or_input_changed", ""
        embedding = obj.get("residue_embedding")
        expected_length = expected_esmc_length(entity, max_sequence_length)
        if (
            not finite_tensor(embedding)
            or embedding.ndim != 2
            or embedding.shape != (expected_length, ESMC_DIM)
            or int(obj.get("used_sequence_length", -1)) != expected_length
        ):
            return False, "invalid_residue_embedding", ""
        return True, "ok", str(tuple(embedding.shape))
    except Exception as exc:
        return False, f"load_failed:{type(exc).__name__}", ""


def validate_esm2_graph(
    path: Path, entity: Entity, expected_configuration_sha256: str
) -> tuple[bool, str, str]:
    try:
        import numpy as np

        if not path.is_file():
            return False, "missing", ""
        with np.load(path, allow_pickle=False) as data:
            required = {"edge_index", "edge_weight", "num_nodes", "meta_json"}
            if not required.issubset(data.files):
                return False, "missing_fields", ""
            edge_index = np.asarray(data["edge_index"])
            edge_weight = np.asarray(data["edge_weight"])
            num_nodes = int(np.asarray(data["num_nodes"]).reshape(-1)[0])
            meta = json.loads(str(np.asarray(data["meta_json"]).reshape(-1)[0]))
        if not common_metadata_valid(
            meta,
            entity,
            "esm2_contact_graph",
            expected_configuration_sha256,
        ):
            return False, "metadata_or_input_changed", ""
        valid = (
            edge_index.ndim == 2 and edge_index.shape[0] == 2
            and edge_weight.ndim == 1 and edge_weight.shape[0] == edge_index.shape[1]
            and num_nodes == len(entity.text) and num_nodes > 0
            and np.isfinite(edge_weight).all()
            and (edge_index.size == 0 or (edge_index.min() >= 0 and edge_index.max() < num_nodes))
        )
        if not valid:
            return False, "invalid_graph", ""
        return True, "ok", f"nodes={num_nodes};edges={edge_index.shape[1]}"
    except Exception as exc:
        return False, f"load_failed:{type(exc).__name__}", ""


def validate_artifact(
    path: Path,
    entity: Entity,
    feature: str,
    args: argparse.Namespace,
) -> tuple[bool, str, str]:
    expected_configuration = configuration_sha256(
        args, entity.dataset, feature
    )
    if feature == "unimol2":
        return validate_unimol(path, entity, expected_configuration)
    if feature == "rdkit":
        return validate_rdkit(path, entity, expected_configuration)
    if feature == "esmc":
        return validate_esmc(
            path,
            entity,
            args.esmc_max_sequence_length,
            expected_configuration,
        )
    if feature == "esm2_contact_graph":
        return validate_esm2_graph(path, entity, expected_configuration)
    raise ValueError(feature)


def manifest_row(
    embedding_root: Path,
    entity: Entity,
    feature: str,
    path: Path,
    status: str,
    detail: str,
    shape: str = "",
) -> dict[str, Any]:
    try:
        relative_path = path.relative_to(embedding_root).as_posix()
    except ValueError:
        relative_path = path.as_posix()
    return {
        "dataset": entity.dataset,
        "entity_type": entity.entity_type,
        "entity_id": entity.entity_id,
        "text_sha256": entity.text_sha256,
        "feature": feature,
        "artifact": relative_path,
        "status": status,
        "detail": detail,
        "shape": shape,
        "updated_at": utc_timestamp(),
    }


MANIFEST_FIELDS = (
    "dataset", "entity_type", "entity_id", "text_sha256", "feature",
    "artifact", "status", "detail", "shape", "updated_at",
)


def atomic_replace_bytes(path: Path, writer: Any, suffix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp{suffix}"
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_torch_save(path: Path, value: Any) -> None:
    import torch

    atomic_replace_bytes(path, lambda temporary: torch.save(value, temporary), ".pt")


def atomic_npz_save(path: Path, **arrays: Any) -> None:
    import numpy as np

    atomic_replace_bytes(
        path,
        lambda temporary: np.savez_compressed(temporary, **arrays),
        ".npz",
    )


def atomic_write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    def writer(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            csv_writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
            csv_writer.writeheader()
            csv_writer.writerows(rows)

    atomic_replace_bytes(path, writer, ".csv")


def atomic_write_json(path: Path, value: Any) -> None:
    def writer(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")

    atomic_replace_bytes(path, writer, ".json")


def write_manifest(directory: Path, rows: Mapping[str, Mapping[str, Any]]) -> None:
    ordered = [rows[key] for key in sorted(rows)]
    atomic_write_csv(directory / "manifest.csv", ordered)


def scan_feature(
    embedding_root: Path,
    entities: Sequence[Entity],
    feature: str,
    force: bool,
    args: argparse.Namespace,
) -> ScanResult:
    rows: dict[str, dict[str, Any]] = {}
    pending: list[Entity] = []
    valid_count = 0
    for entity in entities:
        path = artifact_path(embedding_root, entity, feature)
        valid, detail, shape = validate_artifact(path, entity, feature, args)
        if valid and not force:
            status = "valid"
            valid_count += 1
        else:
            status = "pending"
            pending.append(entity)
            if force and valid:
                detail = "forced_regeneration"
        rows[entity.entity_id] = manifest_row(
            embedding_root, entity, feature, path, status, detail, shape
        )
    return ScanResult(rows, pending, valid_count)


def base_artifact(
    entity: Entity, feature: str, args: argparse.Namespace
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "dataset": entity.dataset,
        "entity_type": entity.entity_type,
        "entity_id": entity.entity_id,
        "feature": feature,
        "text_sha256": entity.text_sha256,
        "configuration": feature_configuration(args, entity.dataset, feature),
        "configuration_sha256": configuration_sha256(
            args, entity.dataset, feature
        ),
        "created_at": utc_timestamp(),
    }


def one_hot_unknown(value: Any, choices: Sequence[Any]) -> list[float]:
    result = [0.0] * (len(choices) + 1)
    try:
        index = list(choices).index(value)
    except ValueError:
        index = len(choices)
    result[index] = 1.0
    return result


def generate_rdkit_artifact(
    entity: Entity, args: argparse.Namespace
) -> dict[str, Any]:
    import numpy as np
    import torch
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, BRICS, MACCSkeys

    mol = Chem.MolFromSmiles(entity.text)
    if mol is None or mol.GetNumAtoms() <= 1:
        raise ValueError("RDKit could not parse a multi-atom molecule")

    if entity.dataset in DTA_DATASETS:
        atom_numbers = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
        degrees = [0, 1, 2, 3, 4, 5]
        charges = [-2, -1, 0, 1, 2]
        hybrids = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]
        chiralities = [0, 1, 2, 3]

        def ue_atom(atom: Any) -> np.ndarray:
            values: list[float] = []
            values += one_hot_unknown(int(atom.GetAtomicNum()), atom_numbers)
            values += one_hot_unknown(int(atom.GetTotalDegree()), degrees)
            values += one_hot_unknown(int(atom.GetFormalCharge()), charges)
            values += one_hot_unknown(str(atom.GetHybridization()), hybrids)
            values += one_hot_unknown(int(atom.GetChiralTag()), chiralities)
            values += [
                float(atom.GetIsAromatic()),
                float(atom.IsInRing()),
                float(atom.GetTotalNumHs(includeNeighbors=True)) / 8.0,
                float(atom.GetMass()) / 200.0,
                float(atom.GetTotalValence()) / 8.0,
            ]
            return np.asarray(values, dtype=np.float32)

        ue_x_array = np.stack([ue_atom(atom) for atom in mol.GetAtoms()])
        ue_adj_array = np.eye(mol.GetNumAtoms(), dtype=np.float32)
        bond_weights = {
            str(Chem.BondType.SINGLE): 1.0,
            str(Chem.BondType.DOUBLE): 1.2,
            str(Chem.BondType.TRIPLE): 1.4,
            str(Chem.BondType.AROMATIC): 1.1,
        }
        for bond in mol.GetBonds():
            source, target = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            weight = bond_weights.get(str(bond.GetBondType()), 1.0)
            ue_adj_array[source, target] = weight
            ue_adj_array[target, source] = weight
        degree = np.maximum(ue_adj_array.sum(axis=-1, keepdims=True), 1e-6)
        ue_adj_array = ue_adj_array / np.sqrt(degree * degree.T)
        artifact = base_artifact(entity, "rdkit", args)
        artifact.update(
            {
                "smiles": entity.text,
                "ue_x": torch.from_numpy(ue_x_array).float(),
                "ue_adj": torch.from_numpy(ue_adj_array).float(),
            }
        )
        return artifact

    fp_dimensions = (2048, 2048, 2048, 167)
    fingerprints = (
        AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=True),
        AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048, useChirality=True),
        Chem.RDKFingerprint(
            mol, fpSize=2048, minPath=1, maxPath=7, nBitsPerHash=2, useHs=False
        ),
        MACCSkeys.GenMACCSKeys(mol),
    )
    fp_parts = []
    for fingerprint, dimension in zip(fingerprints, fp_dimensions):
        array = np.zeros((dimension,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        fp_parts.append(array.astype(np.float32, copy=False))
    fingerprint = torch.from_numpy(np.concatenate(fp_parts))

    atom_numbers = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
    degrees = [0, 1, 2, 3, 4, 5]
    charges = [-2, -1, 0, 1, 2]
    hybrids = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    hydrogen_counts = [0, 1, 2, 3, 4]
    chiralities = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    ]
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    stereos = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
    ]

    def bp_atom(atom: Any) -> np.ndarray:
        values: list[float] = []
        values += one_hot_unknown(atom.GetAtomicNum(), atom_numbers)
        values += one_hot_unknown(atom.GetTotalDegree(), degrees)
        values += one_hot_unknown(atom.GetFormalCharge(), charges)
        values += one_hot_unknown(atom.GetHybridization(), hybrids)
        values += one_hot_unknown(atom.GetTotalNumHs(), hydrogen_counts)
        values += one_hot_unknown(atom.GetChiralTag(), chiralities)
        values += [
            float(atom.GetIsAromatic()), float(atom.IsInRing()),
            float(atom.GetMass() * 0.01), float(atom.GetImplicitValence() * 0.1),
            float(atom.GetTotalValence() * 0.1),
        ]
        return np.asarray(values, dtype=np.float32)

    def bond_feature(bond: Any) -> np.ndarray:
        values: list[float] = []
        values += one_hot_unknown(bond.GetBondType(), bond_types)
        values += one_hot_unknown(bond.GetStereo(), stereos)
        values += [float(bond.GetIsConjugated()), float(bond.IsInRing())]
        return np.asarray(values, dtype=np.float32)

    bp_x = torch.from_numpy(np.stack([bp_atom(atom) for atom in mol.GetAtoms()]))
    sources: list[int] = []
    targets: list[int] = []
    edge_features: list[np.ndarray] = []
    for bond in mol.GetBonds():
        source, target = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        values = bond_feature(bond)
        sources.extend((source, target))
        targets.extend((target, source))
        edge_features.extend((values, values))
    bond_dimension = len(bond_types) + 1 + len(stereos) + 1 + 2
    if edge_features:
        bp_edge_index = torch.tensor([sources, targets], dtype=torch.long)
        bp_edge_attr = torch.from_numpy(np.stack(edge_features))
    else:
        bp_edge_index = torch.empty((2, 0), dtype=torch.long)
        bp_edge_attr = torch.empty((0, bond_dimension), dtype=torch.float32)

    cut_bonds: list[tuple[int, int, np.ndarray]] = []
    for bond_info in BRICS.FindBRICSBonds(mol):
        source, target = map(int, bond_info[0])
        bond = mol.GetBondBetweenAtoms(source, target)
        if bond is not None:
            cut_bonds.append((source, target, bond_feature(bond)))
    editable = Chem.RWMol(mol)
    for source, target, _ in cut_bonds:
        if editable.GetBondBetweenAtoms(source, target) is not None:
            editable.RemoveBond(source, target)
    fragments = Chem.GetMolFrags(editable.GetMol(), asMols=False, sanitizeFrags=False)
    if not fragments:
        fragments = (tuple(range(mol.GetNumAtoms())),)
    atom_to_motif = torch.zeros(mol.GetNumAtoms(), dtype=torch.long)
    for motif_index, atom_ids in enumerate(fragments):
        for atom_id in atom_ids:
            atom_to_motif[int(atom_id)] = motif_index
    motif_sources: list[int] = []
    motif_targets: list[int] = []
    motif_features: list[np.ndarray] = []
    for source, target, values in cut_bonds:
        source_motif = int(atom_to_motif[source])
        target_motif = int(atom_to_motif[target])
        if source_motif != target_motif:
            motif_sources.extend((source_motif, target_motif))
            motif_targets.extend((target_motif, source_motif))
            motif_features.extend((values, values))
    if motif_features:
        motif_edge_index = torch.tensor(
            [motif_sources, motif_targets], dtype=torch.long
        )
        motif_edge_attr = torch.from_numpy(np.stack(motif_features))
    else:
        motif_edge_index = torch.empty((2, 0), dtype=torch.long)
        motif_edge_attr = torch.empty((0, bond_dimension), dtype=torch.float32)
    motif_sizes = torch.bincount(
        atom_to_motif, minlength=len(fragments)
    ).float().unsqueeze(-1)

    artifact = base_artifact(entity, "rdkit", args)
    artifact.update(
        {
            "smiles": entity.text,
            "fingerprint": fingerprint.float(),
            "bp_x": bp_x.float(),
            "bp_edge_index": bp_edge_index,
            "bp_edge_attr": bp_edge_attr.float(),
            "atom_to_motif": atom_to_motif,
            "motif_edge_index": motif_edge_index,
            "motif_edge_attr": motif_edge_attr.float(),
            "motif_sizes": motif_sizes,
        }
    )
    return artifact


def to_numpy(value: Any) -> Any:
    import numpy as np

    if value is None:
        return None
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(value)


def normalize_unimol_cls(raw: Any, count: int) -> list[Any]:
    import numpy as np

    if isinstance(raw, (list, tuple)) and len(raw) == count:
        return [np.asarray(to_numpy(value), dtype=np.float32).reshape(-1) for value in raw]
    array = to_numpy(raw)
    if array is not None and array.ndim == 2 and array.shape[0] == count:
        return [np.asarray(array[index], dtype=np.float32).reshape(-1) for index in range(count)]
    if count == 1 and array is not None:
        return [np.asarray(array, dtype=np.float32).reshape(-1)]
    raise ValueError("Could not parse UniMol2 cls_repr")


def normalize_unimol_atomic(raw: Any, count: int) -> list[Any]:
    import numpy as np

    if isinstance(raw, (list, tuple)) and len(raw) == count:
        values = list(raw)
    else:
        array = to_numpy(raw)
        if array is not None and array.dtype == object and array.ndim == 1 and len(array) == count:
            values = list(array)
        elif array is not None and array.ndim == 3 and array.shape[0] == count:
            values = [array[index] for index in range(count)]
        elif count == 1 and array is not None and array.ndim in (1, 2):
            values = [array]
        else:
            raise ValueError("Could not parse UniMol2 atomic_reprs")
    result = []
    for value in values:
        array = np.asarray(to_numpy(value), dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError(f"Invalid UniMol2 atomic shape: {array.shape}")
        result.append(array)
    return result


def parse_unimol_output(
    output: Any, count: int, need_atomic: bool
) -> tuple[list[Any], list[Any] | None]:
    import numpy as np

    if isinstance(output, dict):
        cls_raw = next(
            (output[key] for key in ("cls_repr", "cls_reprs") if key in output),
            None,
        )
        if cls_raw is None:
            raise ValueError(f"UniMol2 output has no CLS field: {sorted(output)}")
        cls_values = normalize_unimol_cls(cls_raw, count)
        if not need_atomic:
            return cls_values, None
        atomic_raw = next(
            (
                output[key]
                for key in ("atomic_reprs", "atomic_repr", "atom_reprs")
                if key in output
            ),
            None,
        )
        if atomic_raw is None:
            raise ValueError(f"UniMol2 output has no atom field: {sorted(output)}")
        return cls_values, normalize_unimol_atomic(atomic_raw, count)
    if isinstance(output, list) and len(output) == count and all(
        isinstance(item, dict) for item in output
    ):
        cls_values, atomic_values = [], []
        for item in output:
            one_cls, one_atomic = parse_unimol_output(item, 1, need_atomic)
            cls_values.append(one_cls[0])
            if need_atomic and one_atomic is not None:
                atomic_values.append(one_atomic[0])
        return cls_values, atomic_values if need_atomic else None
    if isinstance(output, list) and len(output) == count and not any(
        isinstance(item, dict) for item in output
    ):
        # unimol_tools 0.1.x returns a plain list of per-molecule arrays when
        # return_atomic_reprs=False (DTA datasets: Davis/KIBA/BindingDB).
        cls_values = [
            np.asarray(to_numpy(item), dtype=np.float32).reshape(-1) for item in output
        ]
        return cls_values, None
    array = to_numpy(output)
    if array is not None and array.ndim == 2 and array.shape[0] == count:
        # Bare stacked array of shape (count, dim).
        return (
            [
                np.asarray(array[index], dtype=np.float32).reshape(-1)
                for index in range(count)
            ],
            None,
        )
    raise ValueError(f"Unsupported UniMol2 output type: {type(output)}")


@contextmanager
def model_download_environment(offline: bool, hf_endpoint: str) -> Iterator[None]:
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM", "HF_ENDPOINT")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if hf_endpoint:
            os.environ["HF_ENDPOINT"] = hf_endpoint
        if offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        # Synchronize the cached Hugging Face offline flag before online fallback.
        try:
            import huggingface_hub.constants as hf_constants

            hf_constants.HF_HUB_OFFLINE = bool(offline)
        except ImportError:
            pass
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        try:
            import huggingface_hub.constants as hf_constants

            hf_constants.HF_HUB_OFFLINE = previous["HF_HUB_OFFLINE"] == "1"
        except ImportError:
            pass


def resolve_device(value: str) -> str:
    import torch

    if value == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        print(f"[WARNING] CUDA is unavailable; using CPU instead of {value}")
        return "cpu"
    return value


def load_unimol_model(args: argparse.Namespace) -> Any:
    def load(offline: bool) -> Any:
        with model_download_environment(offline, args.hf_endpoint):
            from unimol_tools import UniMolRepr

            common = {
                "data_type": "molecule",
                "batch_size": args.unimol_batch_size,
                "remove_hs": False,
                "model_name": args.unimol_model_name,
                "model_size": args.unimol_model_size,
            }
            try:
                model = UniMolRepr(**common, device=args.device)
            except TypeError:
                import torch

                model = UniMolRepr(
                    **common,
                    use_cuda=torch.cuda.is_available() and args.device != "cpu",
                )
            try:
                model._project_model_name = args.unimol_model_name
                model._project_model_size = args.unimol_model_size
            except Exception:
                pass
            return model

    if not args.no_offline_first:
        try:
            print("[UniMol2] Loading from the local model cache...")
            return load(True)
        except Exception as exc:
            if args.offline:
                raise
            print(
                f"[UniMol2] Local cache unavailable ({type(exc).__name__}); "
                "downloading the model."
            )
    return load(args.offline)


def generate_unimol_batch(
    model: Any,
    entities: Sequence[Entity],
    need_atomic: bool,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    import numpy as np
    import torch

    output = model.get_repr(
        [entity.text for entity in entities], return_atomic_reprs=need_atomic
    )
    cls_values, atomic_values = parse_unimol_output(output, len(entities), need_atomic)
    artifacts = []
    for index, (entity, cls_value) in enumerate(zip(entities, cls_values)):
        cls_array = np.asarray(cls_value, dtype=np.float32).reshape(-1)
        if cls_array.size != UNIMOL_DIM or not np.isfinite(cls_array).all():
            raise ValueError(
                f"UniMol2 CLS dim/value error for {entity.entity_id}: {cls_array.shape}"
            )
        artifact = base_artifact(entity, "unimol2", args)
        artifact.update(
            {
                "smiles": entity.text,
                "model_name": getattr(model, "_project_model_name", "unimolv2"),
                "model_size": getattr(model, "_project_model_size", "1.1B"),
                "cls_repr": torch.from_numpy(cls_array).to(torch.float16),
            }
        )
        if need_atomic:
            assert atomic_values is not None
            atomic_array = np.asarray(atomic_values[index], dtype=np.float32)
            if (
                atomic_array.ndim != 2
                or atomic_array.shape[1] != UNIMOL_DIM
                or not np.isfinite(atomic_array).all()
            ):
                raise ValueError(
                    f"UniMol2 atom shape/value error for {entity.entity_id}: {atomic_array.shape}"
                )
            artifact["atomic_reprs"] = torch.from_numpy(atomic_array).to(torch.float16)
        artifacts.append(artifact)
    return artifacts


def normalize_esmc_hidden(
    hidden_states: Any, expected_length: int
) -> Any:
    import torch

    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError(f"ESMC hidden_states is {type(hidden_states)}, expected Tensor")
    value = hidden_states.detach().float().cpu()
    if value.ndim == 4:
        if value.shape[1] == 1:
            value = value[-1, 0]
        elif value.shape[0] == 1:
            value = value[0, -1]
        else:
            value = value[-1]
    if value.ndim == 3:
        value = value.squeeze(0) if value.shape[0] == 1 else value[-1]
    if value.ndim != 2 or value.shape[1] != ESMC_DIM:
        raise ValueError(f"Invalid ESMC hidden state shape: {tuple(value.shape)}")
    if value.shape[0] == expected_length + 2:
        value = value[1:-1]
    if value.shape[0] != expected_length:
        raise ValueError(
            f"ESMC residue length {value.shape[0]} != sequence length {expected_length}"
        )
    return value.contiguous()


def load_esmc_model(args: argparse.Namespace) -> Any:
    def load(offline: bool) -> Any:
        with model_download_environment(offline, args.hf_endpoint):
            from esm.models.esmc import ESMC

            model = ESMC.from_pretrained(args.esmc_model_name).to(args.device)
            model.eval()
            return model

    if not args.no_offline_first:
        try:
            print("[ESMC] Loading from the local model cache...")
            return load(True)
        except Exception as exc:
            if args.offline:
                raise
            print(
                f"[ESMC] Local cache unavailable ({type(exc).__name__}); "
                "downloading the model."
            )
    return load(args.offline)


def generate_esmc_artifact(
    model: Any, entity: Entity, args: argparse.Namespace
) -> dict[str, Any]:
    import torch
    from esm.sdk.api import ESMProtein, LogitsConfig

    sequence = entity.text
    if args.esmc_max_sequence_length > 0:
        sequence = sequence[: args.esmc_max_sequence_length]
    protein = ESMProtein(sequence=sequence)
    encoded = model.encode(protein)
    if hasattr(encoded, "to"):
        encoded = encoded.to(args.device)
    with torch.no_grad():
        output = model.logits(
            encoded, LogitsConfig(sequence=True, return_hidden_states=True)
        )
    embedding = normalize_esmc_hidden(output.hidden_states, len(sequence))
    artifact = base_artifact(entity, "esmc", args)
    artifact.update(
        {
            "model_name": args.esmc_model_name,
            "original_sequence_length": len(entity.text),
            "used_sequence_length": len(sequence),
            "truncated": len(sequence) != len(entity.text),
            "residue_embedding": embedding.to(torch.float16),
        }
    )
    return artifact


def trim_contact_map(contact: Any, sequence_length: int) -> Any:
    import numpy as np

    contact = np.asarray(contact, dtype=np.float32)
    if contact.ndim != 2 or contact.shape[0] != contact.shape[1]:
        raise ValueError(f"Contact map is not square: {contact.shape}")
    if contact.shape[0] == sequence_length:
        return contact
    if contact.shape[0] >= sequence_length + 2:
        return contact[1 : sequence_length + 1, 1 : sequence_length + 1]
    if contact.shape[0] > sequence_length:
        return contact[:sequence_length, :sequence_length]
    raise ValueError(f"Contact map {contact.shape} is shorter than {sequence_length}")


class HFESM2Predictor:
    def __init__(self, args: argparse.Namespace):
        import torch
        from transformers import AutoTokenizer, EsmModel

        kwargs: dict[str, Any] = {}
        if args.esm2_torch_dtype == "float16":
            kwargs["torch_dtype"] = torch.float16
        elif args.esm2_torch_dtype == "bfloat16":
            kwargs["torch_dtype"] = torch.bfloat16
        elif args.esm2_torch_dtype == "float32":
            kwargs["torch_dtype"] = torch.float32
        if args.offline:
            kwargs["local_files_only"] = True
        with model_download_environment(args.offline, args.hf_endpoint):
            self.tokenizer = AutoTokenizer.from_pretrained(
                args.esm2_model_name, local_files_only=args.offline
            )
            try:
                self.model = EsmModel.from_pretrained(
                    args.esm2_model_name, attn_implementation="eager", **kwargs
                )
            except TypeError:
                self.model = EsmModel.from_pretrained(args.esm2_model_name, **kwargs)
        self.model.to(args.device).eval()
        self.device = args.device

    def predict(self, sequence: str) -> Any:
        import torch

        encoded = self.tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        with torch.no_grad():
            if hasattr(self.model, "predict_contacts"):
                contact = self.model.predict_contacts(
                    input_ids, attention_mask=attention_mask
                )
            else:
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    return_dict=True,
                )
                if not hasattr(self.model, "contact_head"):
                    raise AttributeError("Transformers ESM2 model has no contact predictor")
                contact = self.model.contact_head(input_ids, output.attentions)
        return trim_contact_map(contact[0].detach().float().cpu().numpy(), len(sequence))


class FairESM2Predictor:
    def __init__(self, args: argparse.Namespace):
        import importlib
        import sys
        import torch

        if args.fair_esm_repo_path:
            repository = str(Path(args.fair_esm_repo_path).resolve())
            if repository not in sys.path:
                sys.path.insert(0, repository)
        import esm

        esm = importlib.reload(esm)
        if not hasattr(esm, "pretrained"):
            raise ImportError(
                "Imported esm package is not Meta fair-esm; use --esm2-backend "
                "transformers or provide --fair-esm-repo-path."
            )
        loader = getattr(esm.pretrained, args.esm2_fair_model_name)
        self.model, alphabet = loader()
        self.model.to(args.device).eval()
        self.converter = alphabet.get_batch_converter()
        self.device = args.device
        self.torch = torch

    def predict(self, sequence: str) -> Any:
        _, _, tokens = self.converter([("protein", sequence)])
        with self.torch.no_grad():
            output = self.model(tokens.to(self.device), return_contacts=True)
        return trim_contact_map(
            output["contacts"][0].detach().float().cpu().numpy(), len(sequence)
        )


def load_esm2_predictor(args: argparse.Namespace) -> Any:
    print(f"[ESM2] Loading {args.esm2_backend} contact predictor...")
    if args.esm2_backend == "transformers":
        if not args.no_offline_first and not args.offline:
            offline_args = copy(args)
            offline_args.offline = True
            try:
                print("[ESM2] Loading from the local model cache...")
                return HFESM2Predictor(offline_args)
            except Exception as exc:
                print(
                    f"[ESM2] Local cache unavailable ({type(exc).__name__}); "
                    "downloading the model."
                )
        return HFESM2Predictor(args)
    return FairESM2Predictor(args)


def sliding_window_starts(length: int, window: int, overlap: int) -> list[int]:
    if length <= window:
        return [0]
    overlap = max(0, min(overlap, window - 1))
    stride = max(1, window - overlap)
    starts = list(range(0, length - window + 1, stride))
    final_start = length - window
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def contact_to_sparse_edges(
    contact: Any, probability_threshold: float, top_k: int, minimum_separation: int
) -> tuple[Any, Any]:
    import numpy as np

    contact = np.maximum(
        np.nan_to_num(np.asarray(contact, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
    )
    length = contact.shape[0]
    edges: dict[tuple[int, int], float] = {}
    if probability_threshold > 0:
        rows, columns = np.where(contact >= probability_threshold)
        for source, target in zip(rows.tolist(), columns.tolist()):
            if source == target or abs(source - target) < minimum_separation:
                continue
            weight = float(contact[source, target])
            edges[(source, target)] = max(weight, edges.get((source, target), 0.0))
            edges[(target, source)] = max(weight, edges.get((target, source), 0.0))
    if top_k > 0:
        for source in range(length):
            scores = contact[source].copy()
            low = max(0, source - minimum_separation + 1)
            high = min(length, source + minimum_separation)
            scores[low:high] = -1.0
            indices = (
                np.argpartition(scores, -top_k)[-top_k:]
                if top_k < length
                else np.arange(length)
            )
            for target in indices[scores[indices] > 0].tolist():
                weight = float(contact[source, target])
                edges[(source, target)] = max(weight, edges.get((source, target), 0.0))
                edges[(target, source)] = max(weight, edges.get((target, source), 0.0))
    if not edges:
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32)
    items = sorted(edges.items())
    edge_index = np.asarray(
        [[source for (source, _), _ in items], [target for (_, target), _ in items]],
        dtype=np.int64,
    )
    edge_weight = np.asarray([weight for _, weight in items], dtype=np.float32)
    return edge_index, edge_weight


def coalesce_edges(edge_index: Any, edge_weight: Any, num_nodes: int) -> tuple[Any, Any]:
    import numpy as np

    edges: dict[tuple[int, int], float] = {}
    for index in range(edge_index.shape[1]):
        source, target = int(edge_index[0, index]), int(edge_index[1, index])
        if 0 <= source < num_nodes and 0 <= target < num_nodes:
            edges[(source, target)] = max(
                float(edge_weight[index]), edges.get((source, target), 0.0)
            )
    if not edges:
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32)
    items = sorted(edges.items())
    return (
        np.asarray(
            [[source for (source, _), _ in items], [target for (_, target), _ in items]],
            dtype=np.int64,
        ),
        np.asarray([weight for _, weight in items], dtype=np.float32),
    )


def generate_esm2_graph(
    entity: Entity, predictor: Any, args: argparse.Namespace
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    sequence = entity.text
    window = args.esm2_window_size
    starts = sliding_window_starts(len(sequence), window, args.esm2_window_overlap)
    edge_indices, edge_weights, windows = [], [], []
    means, maxima = [], []
    for start in starts:
        end = min(len(sequence), start + window)
        contact = predictor.predict(sequence[start:end])
        local_index, local_weight = contact_to_sparse_edges(
            contact,
            args.esm2_probability_threshold,
            args.esm2_top_k_long_range,
            args.esm2_minimum_separation,
        )
        if local_index.shape[1] > 0:
            edge_indices.append(local_index + start)
            edge_weights.append(local_weight)
        means.append(float(np.mean(contact)))
        maxima.append(float(np.max(contact)))
        windows.append({"start": start, "end": end})
    if edge_indices:
        edge_index, edge_weight = coalesce_edges(
            np.concatenate(edge_indices, axis=1),
            np.concatenate(edge_weights),
            len(sequence),
        )
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_weight = np.empty((0,), dtype=np.float32)
    metadata = base_artifact(entity, "esm2_contact_graph", args)
    metadata.update(
        {
            "model_name": (
                args.esm2_model_name
                if args.esm2_backend == "transformers"
                else args.esm2_fair_model_name
            ),
            "backend": args.esm2_backend,
            "num_nodes": len(sequence),
            "window_size": window,
            "window_overlap": args.esm2_window_overlap,
            "window_count": len(windows),
            "windows": windows,
            "full_length_coverage": True,
            "probability_threshold": args.esm2_probability_threshold,
            "top_k_long_range": args.esm2_top_k_long_range,
            "minimum_separation": args.esm2_minimum_separation,
            "mean_contact": float(np.mean(means)),
            "max_contact": float(np.max(maxima)),
            "num_edges_directed": int(edge_index.shape[1]),
        }
    )
    arrays = {
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "num_nodes": np.asarray([len(sequence)], dtype=np.int32),
        "meta_json": np.asarray([json.dumps(metadata, ensure_ascii=False)]),
    }
    return arrays, metadata


def clear_model_memory(value: Any) -> None:
    try:
        del value
    except Exception:
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def progress(items: Sequence[Any], description: str) -> Iterable[Any]:
    try:
        from tqdm import tqdm

        return tqdm(items, desc=description, unit="item")
    except ImportError:
        return items


def chunked(items: Sequence[Entity], size: int) -> Iterable[list[Entity]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def mark_generated(
    args: argparse.Namespace,
    entity: Entity,
    feature: str,
    rows: dict[str, dict[str, Any]],
) -> None:
    path = artifact_path(args.embedding_root, entity, feature)
    valid, detail, shape = validate_artifact(path, entity, feature, args)
    if not valid:
        raise RuntimeError(f"Post-write validation failed: {detail}")
    rows[entity.entity_id] = manifest_row(
        args.embedding_root, entity, feature, path, "valid", detail, shape
    )


def mark_failed(
    args: argparse.Namespace,
    entity: Entity,
    feature: str,
    rows: dict[str, dict[str, Any]],
    exc: Exception,
) -> dict[str, str]:
    message = f"{type(exc).__name__}: {exc}"
    rows[entity.entity_id] = manifest_row(
        args.embedding_root,
        entity,
        feature,
        artifact_path(args.embedding_root, entity, feature),
        "failed",
        message[:1000],
    )
    return {
        "dataset": entity.dataset,
        "entity_type": entity.entity_type,
        "entity_id": entity.entity_id,
        "feature": feature,
        "error": message,
    }


def periodically_write_manifest(
    directory: Path,
    rows: dict[str, dict[str, Any]],
    completed: int,
    checkpoint_every: int,
) -> None:
    if completed % checkpoint_every == 0:
        write_manifest(directory, rows)


def process_rdkit(
    args: argparse.Namespace,
    dataset: str,
    scan: ScanResult,
    failures: list[dict[str, str]],
) -> None:
    directory = feature_directory(args.embedding_root, dataset, "rdkit")
    for completed, entity in enumerate(
        progress(scan.pending, f"{dataset} RDKit"), start=1
    ):
        try:
            atomic_torch_save(
                artifact_path(args.embedding_root, entity, "rdkit"),
                generate_rdkit_artifact(entity, args),
            )
            mark_generated(args, entity, "rdkit", scan.rows)
        except Exception as exc:
            failures.append(mark_failed(args, entity, "rdkit", scan.rows, exc))
        periodically_write_manifest(
            directory, scan.rows, completed, args.checkpoint_every
        )
    write_manifest(directory, scan.rows)


def process_unimol(
    args: argparse.Namespace,
    tasks: list[tuple[str, ScanResult]],
    failures: list[dict[str, str]],
) -> None:
    if not any(scan.pending for _, scan in tasks):
        for dataset, scan in tasks:
            write_manifest(
                feature_directory(args.embedding_root, dataset, "unimol2"),
                scan.rows,
            )
        return
    model = load_unimol_model(args)
    try:
        for dataset, scan in tasks:
            directory = feature_directory(args.embedding_root, dataset, "unimol2")
            need_atomic = dataset in CLASSIFICATION_DATASETS
            completed = 0
            ordered = sorted(scan.pending, key=lambda entity: len(entity.text))
            for batch in progress(
                list(chunked(ordered, args.unimol_batch_size)), f"{dataset} UniMol2"
            ):
                try:
                    artifacts = generate_unimol_batch(
                        model, batch, need_atomic, args
                    )
                    for entity, artifact in zip(batch, artifacts):
                        atomic_torch_save(
                            artifact_path(args.embedding_root, entity, "unimol2"),
                            artifact,
                        )
                        mark_generated(args, entity, "unimol2", scan.rows)
                except Exception as batch_exc:
                    # Fall back to per-molecule generation when batch inference fails.
                    for entity in batch:
                        try:
                            artifact = generate_unimol_batch(
                                model, [entity], need_atomic, args
                            )[0]
                            atomic_torch_save(
                                artifact_path(args.embedding_root, entity, "unimol2"),
                                artifact,
                            )
                            mark_generated(args, entity, "unimol2", scan.rows)
                        except Exception as exc:
                            failures.append(
                                mark_failed(args, entity, "unimol2", scan.rows, exc)
                            )
                    print(
                        "[WARNING] UniMol2 batch inference failed; processing "
                        f"molecules individually: {type(batch_exc).__name__}: {batch_exc}"
                    )
                completed += len(batch)
                periodically_write_manifest(
                    directory, scan.rows, completed, args.checkpoint_every
                )
            write_manifest(directory, scan.rows)
    finally:
        clear_model_memory(model)


def process_esmc(
    args: argparse.Namespace,
    tasks: list[tuple[str, ScanResult]],
    failures: list[dict[str, str]],
) -> None:
    if not any(scan.pending for _, scan in tasks):
        for dataset, scan in tasks:
            write_manifest(
                feature_directory(args.embedding_root, dataset, "esmc"), scan.rows
            )
        return
    model = load_esmc_model(args)
    try:
        for dataset, scan in tasks:
            directory = feature_directory(args.embedding_root, dataset, "esmc")
            for completed, entity in enumerate(
                progress(
                    sorted(scan.pending, key=lambda item: len(item.text)),
                    f"{dataset} ESMC",
                ),
                start=1,
            ):
                try:
                    artifact = generate_esmc_artifact(model, entity, args)
                    atomic_torch_save(
                        artifact_path(args.embedding_root, entity, "esmc"), artifact
                    )
                    mark_generated(args, entity, "esmc", scan.rows)
                except Exception as exc:
                    failures.append(mark_failed(args, entity, "esmc", scan.rows, exc))
                periodically_write_manifest(
                    directory, scan.rows, completed, args.checkpoint_every
                )
            write_manifest(directory, scan.rows)
    finally:
        clear_model_memory(model)


def process_esm2(
    args: argparse.Namespace,
    tasks: list[tuple[str, ScanResult]],
    failures: list[dict[str, str]],
) -> None:
    if not any(scan.pending for _, scan in tasks):
        for dataset, scan in tasks:
            write_manifest(
                feature_directory(
                    args.embedding_root, dataset, "esm2_contact_graph"
                ),
                scan.rows,
            )
        return
    predictor = load_esm2_predictor(args)
    try:
        for dataset, scan in tasks:
            directory = feature_directory(
                args.embedding_root, dataset, "esm2_contact_graph"
            )
            for completed, entity in enumerate(
                progress(
                    sorted(scan.pending, key=lambda item: len(item.text)),
                    f"{dataset} ESM2 graph",
                ),
                start=1,
            ):
                try:
                    arrays, _ = generate_esm2_graph(entity, predictor, args)
                    atomic_npz_save(
                        artifact_path(
                            args.embedding_root, entity, "esm2_contact_graph"
                        ),
                        **arrays,
                    )
                    mark_generated(
                        args, entity, "esm2_contact_graph", scan.rows
                    )
                except Exception as exc:
                    failures.append(
                        mark_failed(
                            args, entity, "esm2_contact_graph", scan.rows, exc
                        )
                    )
                periodically_write_manifest(
                    directory, scan.rows, completed, args.checkpoint_every
                )
            write_manifest(directory, scan.rows)
    finally:
        clear_model_memory(predictor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate resumable BP-NET/UE-AlignNet input features. No arguments "
            "means all features for all five datasets."
        )
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["all"],
        help="B3DB BBBP Davis KIBA BindingDB, or all (default: all)",
    )
    parser.add_argument(
        "--features", nargs="+", default=["all"],
        help="unimol2 rdkit esmc esm2_contact_graph, or all (default: all)",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--embedding-root", type=Path, default=DEFAULT_EMBEDDING_ROOT
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Validate inputs and existing artifacts without generating or writing files",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate selected artifacts even when their validation succeeds",
    )
    parser.add_argument(
        "--max-items", type=int, default=0,
        help="At most N pending entities per dataset/feature; 0 means all",
    )
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--no-offline-first", action="store_true")
    parser.add_argument("--hf-endpoint", default="")

    parser.add_argument("--unimol-model-name", default="unimolv2")
    parser.add_argument("--unimol-model-size", default="1.1B")
    parser.add_argument("--unimol-batch-size", type=int, default=32)

    parser.add_argument("--esmc-model-name", default="esmc_600m")
    parser.add_argument("--esmc-max-sequence-length", type=int, default=4500)

    parser.add_argument(
        "--esm2-backend", choices=("transformers", "fair_esm"),
        default="transformers",
    )
    parser.add_argument(
        "--esm2-model-name", default="facebook/esm2_t33_650M_UR50D"
    )
    parser.add_argument("--esm2-fair-model-name", default="esm2_t33_650M_UR50D")
    parser.add_argument(
        "--esm2-torch-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="float32",
    )
    parser.add_argument("--fair-esm-repo-path", default="")
    parser.add_argument("--esm2-window-size", type=int, default=1022)
    parser.add_argument("--esm2-window-overlap", type=int, default=256)
    parser.add_argument("--esm2-probability-threshold", type=float, default=0.5)
    parser.add_argument("--esm2-top-k-long-range", type=int, default=16)
    parser.add_argument("--esm2-minimum-separation", type=int, default=4)
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if args.max_items < 0:
        raise ValueError("--max-items must be >= 0")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be > 0")
    if args.unimol_batch_size <= 0:
        raise ValueError("--unimol-batch-size must be > 0")
    if args.esmc_max_sequence_length < 0:
        raise ValueError("--esmc-max-sequence-length must be >= 0")
    if args.esm2_window_size <= 0:
        raise ValueError("--esm2-window-size must be > 0")
    if not 0 <= args.esm2_window_overlap < args.esm2_window_size:
        raise ValueError("--esm2-window-overlap must be in [0, window_size)")
    if not 0 <= args.esm2_probability_threshold <= 1:
        raise ValueError("--esm2-probability-threshold must be in [0, 1]")
    if args.esm2_top_k_long_range < 0 or args.esm2_minimum_separation < 1:
        raise ValueError("Invalid ESM2 sparsification arguments")


def require_generation_dependencies(
    args: argparse.Namespace,
    scans: Mapping[tuple[str, str], ScanResult],
) -> None:
    """Check only dependencies needed by artifacts that are actually pending."""
    pending_features = {
        feature for (_dataset, feature), scan in scans.items() if scan.pending
    }
    modules: dict[str, set[str]] = {
        "rdkit": {"numpy", "torch", "rdkit"},
        "unimol2": {"numpy", "torch", "unimol_tools"},
        "esmc": {"torch", "esm"},
        "esm2_contact_graph": {
            "numpy",
            "torch",
            "transformers" if args.esm2_backend == "transformers" else "esm",
        },
    }
    missing: dict[str, list[str]] = {}
    for feature in sorted(pending_features):
        absent = sorted(
            module
            for module in modules[feature]
            if importlib.util.find_spec(module) is None
        )
        if absent:
            missing[feature] = absent
    if missing:
        details = "; ".join(
            f"{feature}: {', '.join(names)}" for feature, names in missing.items()
        )
        raise RuntimeError(
            "Missing dependencies for pending features (no embedding files were "
            f"written): {details}"
        )


def print_scan_summary(
    selected: Sequence[str],
    features: Sequence[str],
    entities_by_dataset: Mapping[str, Mapping[str, list[Entity]]],
    scans: Mapping[tuple[str, str], ScanResult],
) -> None:
    print("\nEmbedding and feature preflight")
    print("=" * 78)
    for dataset in selected:
        counts = entities_by_dataset[dataset]
        print(
            f"{dataset:10s} ligands={len(counts.get('ligand', [])):6d} "
            f"proteins={len(counts.get('protein', [])):6d}"
        )
        for feature in features:
            scan = scans.get((dataset, feature))
            if scan is None:
                continue
            print(
                f"  {feature:22s} valid={scan.valid_count:6d} "
                f"pending={len(scan.pending):6d}"
            )
    print("=" * 78)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.data_root = Path(args.data_root)
    args.embedding_root = Path(args.embedding_root)
    validate_arguments(args)
    selected = normalize_dataset_names(args.datasets)
    selected_features = normalize_feature_names(args.features)

    # Complete preflight first. No output directory is modified until every
    # selected split and entity source has passed validation.
    entities_by_dataset = {
        dataset: load_dataset_entities(args.data_root, dataset) for dataset in selected
    }
    scans: dict[tuple[str, str], ScanResult] = {}
    for dataset in selected:
        for feature in selected_features:
            entities = feature_entities(
                dataset, entities_by_dataset[dataset], feature
            )
            if not entities:
                continue
            scan = scan_feature(
                args.embedding_root,
                entities,
                feature,
                args.force,
                args,
            )
            if args.max_items:
                scan.pending = scan.pending[: args.max_items]
            scans[(dataset, feature)] = scan
    print_scan_summary(
        selected, selected_features, entities_by_dataset, scans
    )
    if args.check_only:
        print("Check-only mode: no files were written.")
        return 0

    require_generation_dependencies(args, scans)
    args.device = resolve_device(args.device)
    failures: list[dict[str, str]] = []

    # Generate deterministic RDKit features before loading shared neural models.
    if "rdkit" in selected_features:
        for dataset in selected:
            scan = scans.get((dataset, "rdkit"))
            if scan is not None:
                process_rdkit(args, dataset, scan, failures)
    if "unimol2" in selected_features:
        process_unimol(
            args,
            [
                (dataset, scans[(dataset, "unimol2")])
                for dataset in selected
                if (dataset, "unimol2") in scans
            ],
            failures,
        )
    if "esmc" in selected_features:
        process_esmc(
            args,
            [
                (dataset, scans[(dataset, "esmc")])
                for dataset in selected
                if (dataset, "esmc") in scans
            ],
            failures,
        )
    if "esm2_contact_graph" in selected_features:
        process_esm2(
            args,
            [
                (dataset, scans[(dataset, "esm2_contact_graph")])
                for dataset in selected
                if (dataset, "esm2_contact_graph") in scans
            ],
            failures,
        )

    # Revalidate all expected artifacts before rebuilding the manifests.
    remaining: list[dict[str, str]] = []
    final_counts: dict[str, dict[str, dict[str, int]]] = {}
    for dataset in selected:
        final_counts[dataset] = {}
        for feature in selected_features:
            entities = feature_entities(
                dataset, entities_by_dataset[dataset], feature
            )
            if not entities:
                continue
            final_scan = scan_feature(
                args.embedding_root,
                entities,
                feature,
                False,
                args,
            )
            write_manifest(
                feature_directory(args.embedding_root, dataset, feature),
                final_scan.rows,
            )
            final_counts[dataset][feature] = {
                "valid": final_scan.valid_count,
                "expected": len(entities),
                "remaining": len(final_scan.pending),
            }
            remaining.extend(
                {
                    "dataset": entity.dataset,
                    "entity_type": entity.entity_type,
                    "entity_id": entity.entity_id,
                    "feature": feature,
                }
                for entity in final_scan.pending
            )

    summary = {
        "created_at": utc_timestamp(),
        "datasets": selected,
        "features": selected_features,
        "device": args.device,
        "counts": final_counts,
        "failure_count": len(failures),
        "failures_preview": failures[:100],
        "remaining_count": len(remaining),
        "remaining_preview": remaining[:100],
    }
    args.embedding_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.embedding_root / "generation_summary.json", summary)
    if failures or remaining:
        print(
            f"Generation incomplete: failures={len(failures)}, "
            f"remaining={len(remaining)}. Re-run the command to resume."
        )
        print("See ./data/embedding/generation_summary.json for details.")
        return 1
    print("All selected embedding/feature artifacts are valid and complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
