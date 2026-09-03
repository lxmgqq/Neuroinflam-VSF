#!/usr/bin/env python3
"""Train BP-NET on the preprocessed B3DB and BBBP datasets.

The script consumes scaffold-disjoint splits and validated feature manifests.
With no arguments, it trains seeds 1-5 for both datasets. Optimization proceeds
through branch training, fusion training, and validation-controlled fine-tuning.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable: Iterable[Any], **_: Any) -> Iterable[Any]:
        return iterable


DATASETS = ("B3DB", "BBBP")
DEFAULT_DATA_ROOT = Path("./data/datasets")
DEFAULT_EMBEDDING_ROOT = Path("./data/embedding")
DEFAULT_MODEL_ROOT = Path("./models/BP-NET")
SCENARIO = "scaffold"

ARCHITECTURE_VERSION = "BP-StagedUnifiedThreeBranch-B3DB-BBBP-v2.0"
VARIANT_NAME = "staged_unified_fusion"
MODEL_DIM = 192
GRAPH_HIDDEN_DIM = MODEL_DIM
GRAPH_LAYERS = 5
READOUT_STEPS = 3
DROPOUT = 0.15
BRANCH_DROPOUT = 0.15
UNIMOL_ATOM_DIM = 1536
ATOM_FEAT_DIM = 47
BOND_FEAT_DIM = 14
FP_DIMS = {
    "morgan_r2": 2048,
    "morgan_r3": 2048,
    "rdkit": 2048,
    "maccs": 167,
}
FP_TOTAL_DIM = sum(FP_DIMS.values())
THRESHOLD = 0.5
VAL_SCORE_WEIGHTS = {"AUPRC": 0.45, "ROC-AUC": 0.35, "MCC": 0.20}

VARIANT_CONFIG = {
    "use_graph": True,
    "use_fingerprint": True,
    "unimol_mode": "atom",
    "atom_attention": False,
    "use_virtual_node": True,
    "use_jk": True,
    "use_ema": True,
    "use_rank": True,
    "branch_dropout": BRANCH_DROPOUT,
}


@dataclass(frozen=True)
class DatasetDefaults:
    branch_max_epochs: int
    branch_patience: int
    fusion_max_epochs: int
    fusion_patience: int
    finetune_max_epochs: int
    finetune_patience: int


DATASET_DEFAULTS = {
    "B3DB": DatasetDefaults(100, 22, 80, 18, 15, 5),
    "BBBP": DatasetDefaults(200, 30, 100, 22, 20, 6),
}


@dataclass
class FeatureItem:
    atomic_reprs: torch.Tensor
    fingerprint: torch.Tensor
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor


@dataclass
class DatasetRuntime:
    dataset: str
    args: argparse.Namespace
    splits: dict[int, dict[str, pd.DataFrame]]
    split_paths: dict[int, dict[str, Path]]
    entity_ids: list[str]
    unimol_index: "ArtifactIndex"
    rdkit_index: "ArtifactIndex"


@dataclass
class RunOutputs:
    probs: np.ndarray
    labels: np.ndarray
    smiles: list[str]
    branch_probs: dict[str, np.ndarray]
    branch_keep_masks: np.ndarray
    fp_weights: np.ndarray


def normalize_entity_id(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            text = str(int(float(text)))
        except ValueError:
            pass
    return text


def entity_id_for_smiles(smiles: str) -> str:
    return hashlib.sha256(str(smiles).encode("utf-8")).hexdigest()


def normalize_datasets(values: Sequence[str]) -> list[str]:
    if not values or any(value.casefold() in {"all", "both"} for value in values):
        return list(DATASETS)
    lookup = {name.casefold(): name for name in DATASETS}
    result: list[str] = []
    for value in values:
        for part in value.split(","):
            key = part.strip().casefold()
            if not key:
                continue
            if key not in lookup:
                raise ValueError(f"Unknown dataset {part!r}; choose B3DB, BBBP, or all")
            if lookup[key] not in result:
                result.append(lookup[key])
    if not result:
        raise ValueError("--datasets cannot be empty")
    return result


def parse_seeds(text: str) -> list[int]:
    result: list[int] = []
    for token in (part.strip() for part in str(text).split(",")):
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"Invalid seed range: {token}")
            result.extend(range(start, end + 1))
        else:
            result.append(int(token))
    result = list(dict.fromkeys(result))
    if not result or any(seed < 0 for seed in result):
        raise ValueError("Seeds must be non-negative integers")
    return result


def hash_mapping(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def torch_load(path: Path, map_location: Any = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # Compatibility fallback when weights_only is unsupported.
        return torch.load(path, map_location=map_location)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except (AttributeError, TypeError):
        pass


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def resolve_device(value: str) -> str:
    if value == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        print(f"[WARNING] CUDA is unavailable; using CPU instead of {value}")
        return "cpu"
    return value


class ArtifactIndex:
    """Map manifest entity IDs to validated artifact paths."""

    def __init__(self, embedding_root: Path, manifest_path: Path, feature: str):
        self.embedding_root = embedding_root.resolve()
        self.manifest_path = manifest_path
        self.feature = feature
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing {feature} manifest: {manifest_path}\n"
                "Run embedding_generation.py first."
            )
        frame = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
        required = {"entity_id", "artifact", "status", "feature", "text_sha256"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{manifest_path} is missing columns: {sorted(missing)}")
        frame = frame[frame["feature"] == feature].copy()
        if frame.empty:
            raise ValueError(f"No {feature} rows in {manifest_path}")
        frame["entity_id"] = frame["entity_id"].map(normalize_entity_id)
        if bool((frame["entity_id"] == "").any()):
            raise ValueError(f"Blank entity_id in {manifest_path}")
        if frame["entity_id"].duplicated().any():
            duplicates = frame.loc[frame["entity_id"].duplicated(), "entity_id"].tolist()
            raise ValueError(f"Duplicate entity IDs in {manifest_path}: {duplicates[:10]}")
        self.paths: dict[str, Path] = {}
        self.rows: dict[str, dict[str, str]] = {}
        for row in frame.to_dict("records"):
            entity_id = normalize_entity_id(row["entity_id"])
            if row["status"] != "valid":
                continue
            artifact = str(row["artifact"]).strip()
            if not artifact:
                raise ValueError(f"Blank artifact path for {entity_id} in {manifest_path}")
            candidate = (self.embedding_root / artifact).resolve()
            try:
                candidate.relative_to(self.embedding_root)
            except ValueError as exc:
                raise ValueError(f"Manifest artifact escapes embedding root: {candidate}") from exc
            self.paths[entity_id] = candidate
            self.rows[entity_id] = {
                "artifact": artifact,
                "text_sha256": str(row["text_sha256"]),
            }

    def require(self, entity_ids: Iterable[str]) -> None:
        normalized = list(dict.fromkeys(normalize_entity_id(value) for value in entity_ids))
        missing_rows = [value for value in normalized if value not in self.paths]
        missing_files = [
            value for value in normalized
            if value in self.paths and not self.paths[value].is_file()
        ]
        if missing_rows or missing_files:
            raise FileNotFoundError(
                f"Incomplete {self.feature} artifacts in {self.manifest_path}: "
                f"missing/invalid rows={len(missing_rows)} examples={missing_rows[:10]}; "
                f"missing files={len(missing_files)} examples={missing_files[:10]}. "
                "Rerun embedding_generation.py to resume generation."
            )

    def path_for(self, entity_id: str) -> Path:
        key = normalize_entity_id(entity_id)
        if key not in self.paths:
            raise KeyError(f"No valid {self.feature} artifact for entity {key}")
        return self.paths[key]

    def fingerprint(self, entity_ids: Iterable[str]) -> str:
        rows = {
            entity_id: self.rows[entity_id]
            for entity_id in sorted(set(entity_ids))
        }
        return hash_mapping(rows)


def manifest_path(root: Path, dataset: str, feature: str) -> Path:
    return root / dataset / "ligand" / feature / "manifest.csv"


def split_paths(data_root: Path, dataset: str, seed: int) -> dict[str, Path]:
    directory = data_root / dataset / "splits" / f"seed_{seed:04d}" / SCENARIO
    return {split: directory / f"{split}.csv" for split in ("train", "val", "test")}


def load_split_bundle(paths: Mapping[str, Path]) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    seen: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        path = paths[split]
        if not path.is_file():
            raise FileNotFoundError(f"Missing split file: {path}")
        frame = pd.read_csv(path, dtype={"SMILES": str}, keep_default_na=False)
        required = {"SMILES", "label", "Split"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frame = frame[["SMILES", "label", "Split"]].copy()
        frame["SMILES"] = frame["SMILES"].astype(str).str.strip()
        frame["label"] = pd.to_numeric(frame["label"], errors="raise")
        frame["Split"] = frame["Split"].astype(str).str.strip().str.lower()
        if frame.empty:
            raise ValueError(f"Empty split file: {path}")
        if bool((frame["SMILES"] == "").any()):
            raise ValueError(f"Blank SMILES in {path}")
        if not bool(frame["label"].isin([0, 1]).all()):
            raise ValueError(f"Labels outside {{0,1}} in {path}")
        if not bool((frame["Split"] == split).all()):
            raise ValueError(f"Split column in {path} must contain only {split!r}")
        if frame["SMILES"].duplicated().any():
            raise ValueError(f"Duplicate SMILES in {path}")
        frame["label"] = frame["label"].astype(int)
        seen[split] = set(frame["SMILES"])
        result[split] = frame.reset_index(drop=True)
    overlaps = {
        "train/val": seen["train"] & seen["val"],
        "train/test": seen["train"] & seen["test"],
        "val/test": seen["val"] & seen["test"],
    }
    leaked = {name: sorted(values)[:5] for name, values in overlaps.items() if values}
    if leaked:
        raise ValueError(f"SMILES leakage across splits: {leaked}")
    return result


def logical_test_hash(frame: pd.DataFrame) -> str:
    ordered = frame[["SMILES", "label"]].sort_values("SMILES").reset_index(drop=True)
    return hashlib.sha256(ordered.to_csv(index=False).encode("utf-8")).hexdigest()


def resolve_dataset_args(base: argparse.Namespace, dataset: str) -> argparse.Namespace:
    args = copy.copy(base)
    defaults = DATASET_DEFAULTS[dataset]
    args.dataset = dataset
    for name in (
        "branch_max_epochs", "branch_patience", "fusion_max_epochs",
        "fusion_patience", "finetune_patience",
    ):
        value = getattr(base, name)
        setattr(args, name, value if value is not None else getattr(defaults, name))
    args.finetune_max_epochs = (
        base.finetune_max_epochs
        if base.finetune_max_epochs is not None
        else defaults.finetune_max_epochs
    )
    args.dataset_model_root = args.model_root / dataset / args.run_name
    return args


def preflight_dataset(dataset: str, args: argparse.Namespace) -> DatasetRuntime:
    bundles: dict[int, dict[str, pd.DataFrame]] = {}
    paths_by_seed: dict[int, dict[str, Path]] = {}
    entity_ids: set[str] = set()
    expected_test_hash: str | None = None
    rows = 0
    for seed in args.seeds:
        paths = split_paths(args.data_root, dataset, seed)
        bundle = load_split_bundle(paths)
        current_test_hash = logical_test_hash(bundle["test"])
        if expected_test_hash is None:
            expected_test_hash = current_test_hash
        elif current_test_hash != expected_test_hash:
            raise ValueError(f"{dataset} fixed test set differs between seeds")
        for frame in bundle.values():
            entity_ids.update(frame["SMILES"].map(entity_id_for_smiles))
            rows += len(frame)
        bundles[seed], paths_by_seed[seed] = bundle, paths

    unimol_index = ArtifactIndex(
        args.embedding_root,
        manifest_path(args.embedding_root, dataset, "unimol2"),
        "unimol2",
    )
    rdkit_index = ArtifactIndex(
        args.embedding_root,
        manifest_path(args.embedding_root, dataset, "rdkit"),
        "rdkit",
    )
    ordered_ids = sorted(entity_ids)
    unimol_index.require(ordered_ids)
    rdkit_index.require(ordered_ids)
    for index in (unimol_index, rdkit_index):
        mismatched = [
            entity_id for entity_id in ordered_ids
            if index.rows[entity_id]["text_sha256"] != entity_id
        ]
        if mismatched:
            raise ValueError(
                f"{index.feature} manifest does not match split SMILES: {mismatched[:10]}"
            )
    print(
        f"[Preflight] {dataset}: split_rows={rows:,}, seeds={len(args.seeds)}, "
        f"molecules={len(ordered_ids):,}"
    )
    return DatasetRuntime(
        dataset, args, bundles, paths_by_seed, ordered_ids, unimol_index, rdkit_index
    )


class FeatureStore:
    def __init__(
        self,
        unimol_index: ArtifactIndex,
        rdkit_index: ArtifactIndex,
        max_cache_items: int,
    ):
        self.unimol_index = unimol_index
        self.rdkit_index = rdkit_index
        self.max_cache_items = max(1, int(max_cache_items))
        self.cache: OrderedDict[str, FeatureItem] = OrderedDict()
        self.keep_all = False

    @staticmethod
    def _tensor(value: Any, dtype: torch.dtype) -> torch.Tensor:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        return tensor.detach().cpu().to(dtype=dtype)

    def _load(self, entity_id: str) -> FeatureItem:
        unimol = torch_load(self.unimol_index.path_for(entity_id))
        graph = torch_load(self.rdkit_index.path_for(entity_id))
        if not isinstance(unimol, dict) or "atomic_reprs" not in unimol:
            raise ValueError(f"Invalid UniMol2 atom artifact for {entity_id}")
        graph_required = {"fingerprint", "bp_x", "bp_edge_index", "bp_edge_attr"}
        if not isinstance(graph, dict) or not graph_required.issubset(graph):
            raise ValueError(f"Invalid BP-NET RDKit artifact for {entity_id}")

        atomic = self._tensor(unimol["atomic_reprs"], torch.float16)
        fingerprint = self._tensor(graph["fingerprint"], torch.float32).reshape(-1)
        x = self._tensor(graph["bp_x"], torch.float32)
        edge_index = self._tensor(graph["bp_edge_index"], torch.long)
        edge_attr = self._tensor(graph["bp_edge_attr"], torch.float32)
        if atomic.ndim != 2 or atomic.shape[0] <= 0 or atomic.shape[1] != UNIMOL_ATOM_DIM:
            raise ValueError(f"UniMol2 atom shape mismatch for {entity_id}: {tuple(atomic.shape)}")
        if fingerprint.numel() != FP_TOTAL_DIM:
            raise ValueError(f"Fingerprint shape mismatch for {entity_id}: {tuple(fingerprint.shape)}")
        if x.ndim != 2 or x.shape[0] <= 1 or x.shape[1] != ATOM_FEAT_DIM:
            raise ValueError(f"Atom graph shape mismatch for {entity_id}: {tuple(x.shape)}")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index shape mismatch for {entity_id}: {tuple(edge_index.shape)}")
        if edge_attr.ndim != 2 or edge_attr.shape != (edge_index.shape[1], BOND_FEAT_DIM):
            raise ValueError(f"edge_attr shape mismatch for {entity_id}: {tuple(edge_attr.shape)}")
        if edge_index.numel() and (
            int(edge_index.min()) < 0 or int(edge_index.max()) >= int(x.shape[0])
        ):
            raise ValueError(f"Graph edge endpoint outside atom range for {entity_id}")
        for name, tensor in (
            ("atomic_reprs", atomic), ("fingerprint", fingerprint),
            ("bp_x", x), ("bp_edge_attr", edge_attr),
        ):
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} contains NaN/Inf for {entity_id}")
        return FeatureItem(atomic, fingerprint, x, edge_index, edge_attr)

    def get(self, entity_id: str) -> FeatureItem:
        key = normalize_entity_id(entity_id)
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        value = self._load(key)
        self.cache[key] = value
        if not self.keep_all:
            while len(self.cache) > self.max_cache_items:
                self.cache.popitem(last=False)
        return value

    def preload(self, entity_ids: Sequence[str], description: str) -> None:
        self.keep_all = True
        for entity_id in tqdm(entity_ids, desc=description):
            self.get(entity_id)


class BBBUnifiedDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, feature_store: FeatureStore):
        self.smiles = frame["SMILES"].astype(str).tolist()
        self.labels = frame["label"].to_numpy(dtype=np.float32)
        self.entity_ids = [entity_id_for_smiles(smiles) for smiles in self.smiles]
        self.feature_store = feature_store

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.feature_store.get(self.entity_ids[index])
        return {
            "smiles": self.smiles[index],
            "label": self.labels[index],
            "atomic_reprs": item.atomic_reprs,
            "fingerprint": item.fingerprint,
            "x": item.x,
            "edge_index": item.edge_index,
            "edge_attr": item.edge_attr,
        }


def collate_unified(batch):
    smiles = [x["smiles"] for x in batch]
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.float32)
    atom_lengths = torch.tensor(
        [int(x["atomic_reprs"].shape[0]) for x in batch], dtype=torch.long
    )
    max_atoms = int(atom_lengths.max().item())
    atom_repr_dim = int(batch[0]["atomic_reprs"].shape[1])
    atomic_reprs = torch.zeros(
        (len(batch), max_atoms, atom_repr_dim), dtype=torch.float32
    )
    atomic_mask = torch.zeros((len(batch), max_atoms), dtype=torch.bool)
    for idx, item in enumerate(batch):
        atom_matrix = item["atomic_reprs"].to(torch.float32)
        if atom_matrix.ndim != 2 or atom_matrix.size(1) != atom_repr_dim:
            raise RuntimeError(
                f"Inconsistent atomic_reprs shape in batch: {tuple(atom_matrix.shape)}"
            )
        n_atoms = atom_matrix.size(0)
        atomic_reprs[idx, :n_atoms] = atom_matrix
        atomic_mask[idx, :n_atoms] = True
    fingerprint = torch.stack([x["fingerprint"] for x in batch], dim=0)

    xs = []
    edge_indices = []
    edge_attrs = []
    batch_index = []
    node_offset = 0
    for i, item in enumerate(batch):
        x = item["x"].to(torch.float32)
        ei = item["edge_index"].to(torch.long)
        ea = item["edge_attr"].to(torch.float32)
        n = x.size(0)
        xs.append(x)
        batch_index.append(torch.full((n,), i, dtype=torch.long))
        if ei.numel() > 0:
            edge_indices.append(ei + node_offset)
            edge_attrs.append(ea)
        node_offset += n

    graph_x = torch.cat(xs, dim=0)
    graph_batch = torch.cat(batch_index, dim=0)
    if len(edge_indices) == 0:
        graph_edge_index = torch.empty((2, 0), dtype=torch.long)
        graph_edge_attr = torch.empty((0, BOND_FEAT_DIM), dtype=torch.float32)
    else:
        graph_edge_index = torch.cat(edge_indices, dim=1)
        graph_edge_attr = torch.cat(edge_attrs, dim=0)
    return {
        "smiles": smiles,
        "labels": labels,
        "atomic_reprs": atomic_reprs,
        "atomic_mask": atomic_mask,
        "atom_lengths": atom_lengths,
        "fingerprint": fingerprint,
        "graph_x": graph_x,
        "graph_edge_index": graph_edge_index,
        "graph_edge_attr": graph_edge_attr,
        "graph_batch": graph_batch,
        "num_graphs": len(batch),
    }


def make_data_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    args: argparse.Namespace,
) -> DataLoader:
    options: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "collate_fn": collate_unified,
    }
    if args.num_workers > 0:
        options["prefetch_factor"] = args.prefetch_factor
        options["persistent_workers"] = args.persistent_workers
    return DataLoader(dataset, **options)


# BP-NET architecture.
def segment_softmax(src, index, num_segments, eps=1e-12):
    """Apply softmax independently to segments indexed from 0 to num_segments - 1."""
    if src.numel() == 0:
        return src

    if hasattr(torch.Tensor, "scatter_reduce_"):
        max_values = torch.full(
            (num_segments,),
            -1e9,
            dtype=src.dtype,
            device=src.device,
        )
        max_values.scatter_reduce_(0, index, src, reduce="amax", include_self=True)
        exp = torch.exp(src - max_values[index])
    else:
        exp = torch.zeros_like(src)
        for i in torch.unique(index):
            mask = index == i
            exp[mask] = torch.exp(src[mask] - src[mask].max())

    denom = torch.zeros(num_segments, dtype=src.dtype, device=src.device)
    denom.index_add_(0, index, exp)
    return exp / (denom[index] + eps)


def segment_mean(x, index, num_segments):
    out = torch.zeros(num_segments, x.size(-1), dtype=x.dtype, device=x.device)
    count = torch.zeros(num_segments, 1, dtype=x.dtype, device=x.device)
    out.index_add_(0, index, x)
    count.index_add_(0, index, torch.ones(x.size(0), 1, dtype=x.dtype, device=x.device))
    return out / count.clamp_min(1.0)


def segment_max(x, index, num_segments):
    if hasattr(torch.Tensor, "scatter_reduce_"):
        out = torch.full(
            (num_segments, x.size(-1)), -1e9, dtype=x.dtype, device=x.device
        )
        expanded = index.unsqueeze(-1).expand_as(x)
        out.scatter_reduce_(0, expanded, x, reduce="amax", include_self=True)
        return torch.where(out < -1e8, torch.zeros_like(out), out)
    rows = []
    for i in range(num_segments):
        mask = index == i
        rows.append(x[mask].max(dim=0).values if mask.any() else x.new_zeros(x.size(-1)))
    return torch.stack(rows, dim=0)


class EdgeAttentionLayer(nn.Module):
    """Bond-aware local message-passing layer."""

    def __init__(self, hidden_dim, edge_dim, dropout=0.15):
        super().__init__()
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.msg_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attn_score = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.LeakyReLU(0.2), nn.Linear(hidden_dim, 1)
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge_index, edge_attr):
        if edge_index.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            e = self.edge_proj(edge_attr)
            h_src, h_dst = h[src], h[dst]
            msg = self.msg_proj(torch.cat([h_src, e], dim=-1))
            score = self.attn_score(torch.cat([h_dst, h_src, e], dim=-1)).squeeze(-1)
            alpha = segment_softmax(score, dst, h.size(0))
            aggr = torch.zeros_like(h)
            aggr.index_add_(0, dst, alpha.unsqueeze(-1) * msg)
            h = self.norm(h + self.dropout(self.gru(aggr, h)))
        else:
            h = self.norm(h)

        return h


class AttentiveGraphReadout(nn.Module):
    def __init__(self, hidden_dim, steps=3, dropout=0.15):
        super().__init__()
        self.steps = steps
        self.context_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU()
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LeakyReLU(0.2), nn.Linear(hidden_dim, 1)
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, batch_index, num_graphs):
        context = self.context_proj(segment_mean(h, batch_index, num_graphs))
        for _ in range(self.steps):
            score = self.attn(torch.cat([h, context[batch_index]], dim=-1)).squeeze(-1)
            alpha = segment_softmax(score, batch_index, num_graphs)
            readout = torch.zeros(num_graphs, h.size(-1), dtype=h.dtype, device=h.device)
            readout.index_add_(0, batch_index, alpha.unsqueeze(-1) * h)
            context = self.norm(context + self.dropout(self.gru(readout, context)))
        return context


class GraphBranch(nn.Module):
    """Graph encoder with edge attention, a virtual node, and multi-statistic pooling."""

    def __init__(self, atom_dim, edge_dim, hidden_dim=192, dropout=0.15):
        super().__init__()
        self.atom_encoder = nn.Sequential(
            nn.LayerNorm(atom_dim), nn.Linear(atom_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList([
            EdgeAttentionLayer(hidden_dim, edge_dim, dropout)
            for _ in range(GRAPH_LAYERS)
        ])
        self.virtual_token = nn.Parameter(torch.zeros(1, hidden_dim))
        nn.init.normal_(self.virtual_token, std=0.02)
        self.virtual_to_atom = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
            for _ in range(GRAPH_LAYERS)
        ])
        self.virtual_gru = nn.ModuleList([
            nn.GRUCell(hidden_dim, hidden_dim) for _ in range(GRAPH_LAYERS)
        ])
        self.readout = AttentiveGraphReadout(hidden_dim, READOUT_STEPS, dropout)
        self.jk_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * GRAPH_LAYERS),
            nn.Linear(hidden_dim * GRAPH_LAYERS, hidden_dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim * 5), nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self, x, edge_index, edge_attr, batch_index, num_graphs,
        use_virtual_node=True, use_jk=True,
    ):
        h = self.atom_encoder(x)
        virtual = self.virtual_token.expand(num_graphs, -1)
        layer_states = []
        for layer_idx, layer in enumerate(self.layers):
            if use_virtual_node:
                h = h + self.virtual_to_atom[layer_idx](virtual)[batch_index]
            h = layer(h, edge_index, edge_attr)
            pooled = segment_mean(h, batch_index, num_graphs)
            if use_virtual_node:
                virtual = self.virtual_gru[layer_idx](pooled, virtual)
            layer_states.append(pooled)

        attentive = self.readout(h, batch_index, num_graphs)
        mean_pool = segment_mean(h, batch_index, num_graphs)
        max_pool = segment_max(h, batch_index, num_graphs)
        jk = self.jk_proj(torch.cat(layer_states, dim=-1)) if use_jk else layer_states[-1]
        virtual_out = virtual if use_virtual_node else torch.zeros_like(mean_pool)
        return self.out(torch.cat([attentive, mean_pool, max_pool, jk, virtual_out], dim=-1))


class FingerprintBranch(nn.Module):
    """Separate encoders preserve complementary fingerprint semantics."""

    def __init__(self, hidden_dim, dropout=0.15):
        super().__init__()
        self.names = list(FP_DIMS.keys())
        self.dims = list(FP_DIMS.values())
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim),
            )
            for dim in self.dims
        ])
        self.view_score = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(), nn.Linear(hidden_dim // 2, 1),
        )
        self.concat_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * len(self.dims)),
            nn.Linear(hidden_dim * len(self.dims), hidden_dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, fingerprint):
        parts = torch.split(fingerprint, self.dims, dim=-1)
        views = torch.stack(
            [encoder(part) for encoder, part in zip(self.encoders, parts)], dim=1
        )
        scores = self.view_score(views).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        weighted = torch.sum(weights.unsqueeze(-1) * views, dim=1)
        concat_h = self.concat_proj(views.reshape(views.size(0), -1))
        return self.norm(weighted + concat_h), weights


class UniMolAtomBranch(nn.Module):
    """Pool variable-length UniMol2 atom embeddings into one molecular vector.

    Frozen atom representations are summarized by mean and max pooling.
    """

    def __init__(self, atom_repr_dim, hidden_dim, dropout=0.15):
        super().__init__()
        self.atom_encoder = nn.Sequential(
            nn.LayerNorm(atom_repr_dim),
            nn.Linear(atom_repr_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, atomic_reprs, atomic_mask, use_attention=False):
        if atomic_reprs.ndim != 3 or atomic_mask.ndim != 2:
            raise RuntimeError(
                f"Invalid UniMol2 atom batch shape: {tuple(atomic_reprs.shape)}, "
                f"mask={tuple(atomic_mask.shape)}"
            )
        if not bool(atomic_mask.any(dim=1).all()):
            raise RuntimeError("The batch contains a molecule with no valid atoms")

        h = self.atom_encoder(atomic_reprs)
        mask_f = atomic_mask.unsqueeze(-1).to(h.dtype)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        mean_pool = (h * mask_f).sum(dim=1) / denom

        neg_large = torch.finfo(h.dtype).min
        max_pool = h.masked_fill(~atomic_mask.unsqueeze(-1), neg_large).max(dim=1).values

        # This non-attentive readout does not produce atom-level attention weights.
        return self.out(torch.cat([mean_pool, max_pool], dim=-1)), None


class ControlledFeatureFusion(nn.Module):
    """Feature-level fusion with training-only branch dropout.

    Dropout is sampled independently for each sample and branch and affects only
    the representations passed to the fusion network.
    """

    def __init__(self, hidden_dim, dropout=0.15, branch_dropout=0.15):
        super().__init__()
        if not 0.0 <= branch_dropout < 1.0:
            raise ValueError("branch_dropout must satisfy 0 <= p < 1")
        self.hidden_dim = int(hidden_dim)
        self.branch_dropout = float(branch_dropout)
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.final_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _sample_keep_mask(self, batch_size, device, dtype):
        if (not self.training) or self.branch_dropout <= 0.0:
            return torch.ones((batch_size, 3), device=device, dtype=dtype)

        keep = (
            torch.rand((batch_size, 3), device=device) >= self.branch_dropout
        )
        empty_rows = ~keep.any(dim=1)
        if bool(empty_rows.any()):
            empty_indices = empty_rows.nonzero(as_tuple=False).squeeze(-1)
            chosen = torch.randint(0, 3, (empty_indices.numel(),), device=device)
            keep[empty_indices, chosen] = True
        return keep.to(dtype=dtype)

    def forward(self, reps):
        if len(reps) != 3:
            raise RuntimeError(
                f"Feature fusion requires three branch representations; received {len(reps)}"
            )
        stacked = torch.stack(reps, dim=1)  # [B, 3, D]
        keep_mask = self._sample_keep_mask(
            stacked.size(0), stacked.device, stacked.dtype
        )
        if self.training and self.branch_dropout > 0.0:
            scale = 1.0 / (1.0 - self.branch_dropout)
        else:
            scale = 1.0
        masked = stacked * keep_mask.unsqueeze(-1) * scale
        fused_repr = self.fusion_mlp(masked.reshape(masked.size(0), -1))
        final_logit = self.final_head(fused_repr).squeeze(-1)
        return final_logit, fused_repr, keep_mask


class BPUnifiedThreeBranch(nn.Module):
    BRANCH_NAMES = ["graph", "fingerprint", "unimol"]
    FP_VIEW_NAMES = list(FP_DIMS.keys())

    def __init__(self, unimol_atom_dim, atom_dim, edge_dim, variant_config):
        super().__init__()
        self.variant_config = dict(variant_config)
        self.graph_branch = GraphBranch(atom_dim, edge_dim, MODEL_DIM, DROPOUT)
        self.fp_branch = FingerprintBranch(MODEL_DIM, DROPOUT)
        self.unimol_atom_branch = UniMolAtomBranch(
            unimol_atom_dim, MODEL_DIM, DROPOUT
        )
        self.graph_head = nn.Linear(MODEL_DIM, 1)
        self.fp_head = nn.Linear(MODEL_DIM, 1)
        self.unimol_head = nn.Linear(MODEL_DIM, 1)
        self.fusion = ControlledFeatureFusion(
            MODEL_DIM,
            dropout=DROPOUT,
            branch_dropout=variant_config.get("branch_dropout", BRANCH_DROPOUT),
        )

    def forward(
        self, atomic_reprs, atomic_mask, fingerprint,
        graph_x, graph_edge_index,
        graph_edge_attr, graph_batch, num_graphs,
    ):
        cfg = self.variant_config
        h_graph = self.graph_branch(
            graph_x, graph_edge_index, graph_edge_attr, graph_batch, num_graphs,
            use_virtual_node=cfg["use_virtual_node"], use_jk=cfg["use_jk"],
        )
        h_fp, fp_weights = self.fp_branch(fingerprint)
        h_unimol, atom_weights = self.unimol_atom_branch(
            atomic_reprs, atomic_mask, use_attention=False,
        )
        if atom_weights is None:
            atom_weights = atomic_reprs.new_full(atomic_mask.shape, float("nan"))

        # Auxiliary losses supervise every branch before fusion dropout is applied.
        logit_graph = self.graph_head(h_graph).squeeze(-1)
        logit_fp = self.fp_head(h_fp).squeeze(-1)
        logit_unimol = self.unimol_head(h_unimol).squeeze(-1)
        logit, fused_repr, branch_keep_mask = self.fusion(
            [h_graph, h_fp, h_unimol]
        )
        return {
            "logit": logit,
            "logit_graph": logit_graph,
            "logit_fingerprint": logit_fp,
            "logit_unimol": logit_unimol,
            "fp_weights": fp_weights,
            "atom_weights": atom_weights,
            "branch_keep_mask": branch_keep_mask,
            "fused_repr": fused_repr,
        }


class ModelEMA:
    def __init__(self, model, decay=0.995, warmup_steps=100):
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.decay = float(decay)
        self.warmup_steps = int(warmup_steps)
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        warm = self.updates / max(self.warmup_steps, 1)
        decay = min(self.decay, self.decay * warm)
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for key, ema_value in ema_state.items():
            source = model_state[key].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(decay).add_(source, alpha=1.0 - decay)
            else:
                ema_value.copy_(source)

    def checkpoint_state(self) -> dict[str, Any]:
        return {"model": cpu_module_state(self.module), "updates": self.updates}

    def load_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        self.module.load_state_dict(state["model"])
        self.updates = int(state["updates"])


def compute_specificity(y_true: np.ndarray, y_pred_class: np.ndarray) -> float:
    matrix = confusion_matrix(y_true, y_pred_class, labels=[0, 1])
    true_negative, false_positive, _, _ = matrix.ravel()
    denominator = true_negative + false_positive
    return float(true_negative / denominator) if denominator > 0 else 0.0


def evaluate_binary_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = THRESHOLD,
) -> dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_pred_proba = np.asarray(y_pred_proba).reshape(-1).astype(float)
    y_pred_class = (y_pred_proba >= threshold).astype(int)
    multiple_classes = len(np.unique(y_true)) > 1
    return {
        "ROC-AUC": float(roc_auc_score(y_true, y_pred_proba)) if multiple_classes else float("nan"),
        "AUPRC": float(average_precision_score(y_true, y_pred_proba)) if multiple_classes else float("nan"),
        "Accuracy": float(accuracy_score(y_true, y_pred_class)),
        "F1-score": float(f1_score(y_true, y_pred_class, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred_class, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred_class, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred_class)) if multiple_classes else float("nan"),
        "Specificity": compute_specificity(y_true, y_pred_class),
        "Threshold": float(threshold),
    }


def compute_val_score(metrics: Mapping[str, Any]) -> float:
    auprc = float(metrics.get("AUPRC", float("nan")))
    roc_auc = float(metrics.get("ROC-AUC", float("nan")))
    mcc = float(metrics.get("MCC", float("nan")))
    auprc = auprc if np.isfinite(auprc) else 0.0
    roc_auc = roc_auc if np.isfinite(roc_auc) else 0.0
    mcc = mcc if np.isfinite(mcc) else -1.0
    return float(
        VAL_SCORE_WEIGHTS["AUPRC"] * auprc
        + VAL_SCORE_WEIGHTS["ROC-AUC"] * roc_auc
        + VAL_SCORE_WEIGHTS["MCC"] * mcc
    )


def pairwise_ranking_loss(logits, labels, margin=0.2, max_pos=64, max_neg=64):
    labels = labels.view(-1).long()
    logits = logits.view(-1)
    pos = logits[labels == 1]
    neg = logits[labels == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return logits.new_tensor(0.0)
    if pos.numel() > max_pos:
        pos = pos[torch.randperm(pos.numel(), device=pos.device)[:max_pos]]
    if neg.numel() > max_neg:
        neg = neg[torch.randperm(neg.numel(), device=neg.device)[:max_neg]]
    return F.relu(margin - pos[:, None] + neg[None, :]).mean()


BRANCH_TO_LOGIT = {
    "graph": "logit_graph",
    "fingerprint": "logit_fingerprint",
    "unimol": "logit_unimol",
}


def batch_forward(model: nn.Module, batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return model(
        atomic_reprs=batch["atomic_reprs"].to(device, non_blocking=True),
        atomic_mask=batch["atomic_mask"].to(device, non_blocking=True),
        fingerprint=batch["fingerprint"].to(device, non_blocking=True),
        graph_x=batch["graph_x"].to(device, non_blocking=True),
        graph_edge_index=batch["graph_edge_index"].to(device, non_blocking=True),
        graph_edge_attr=batch["graph_edge_attr"].to(device, non_blocking=True),
        graph_batch=batch["graph_batch"].to(device, non_blocking=True),
        num_graphs=batch["num_graphs"],
    )


@torch.no_grad()
def run_eval(model: nn.Module, loader: DataLoader, device: torch.device) -> RunOutputs:
    model.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_smiles: list[str] = []
    all_branch_probs: dict[str, list[np.ndarray]] = {
        "graph": [], "fingerprint": [], "unimol": []
    }
    all_branch_keep_masks: list[np.ndarray] = []
    all_fp_weights: list[np.ndarray] = []
    for batch in loader:
        output = batch_forward(model, batch, device)
        all_probs.append(torch.sigmoid(output["logit"]).cpu().numpy())
        all_labels.append(batch["labels"].cpu().numpy())
        all_smiles.extend(batch["smiles"])
        for name in all_branch_probs:
            all_branch_probs[name].append(
                torch.sigmoid(output[f"logit_{name}"]).cpu().numpy()
            )
        all_branch_keep_masks.append(output["branch_keep_mask"].cpu().numpy())
        all_fp_weights.append(output["fp_weights"].cpu().numpy())
    if not all_probs:
        raise ValueError("Evaluation loader is empty")
    return RunOutputs(
        probs=np.concatenate(all_probs),
        labels=np.concatenate(all_labels).astype(int),
        smiles=all_smiles,
        branch_probs={name: np.concatenate(values) for name, values in all_branch_probs.items()},
        branch_keep_masks=np.concatenate(all_branch_keep_masks),
        fp_weights=np.concatenate(all_fp_weights),
    )


def prediction_frame(outputs: RunOutputs) -> pd.DataFrame:
    data: dict[str, Any] = {
        "SMILES": outputs.smiles,
        "y_true": outputs.labels,
        "y_pred_proba": outputs.probs,
        "y_pred_class": (outputs.probs >= THRESHOLD).astype(int),
    }
    for name, values in outputs.branch_probs.items():
        data[f"auxiliary_branch_prob_{name}"] = values
    for index, name in enumerate(BPUnifiedThreeBranch.BRANCH_NAMES):
        data[f"fusion_input_enabled_{name}"] = outputs.branch_keep_masks[:, index]
    for index, name in enumerate(BPUnifiedThreeBranch.FP_VIEW_NAMES):
        data[f"fingerprint_weight_{name}"] = outputs.fp_weights[:, index]
    return pd.DataFrame(data)


def summarize_diagnostics(outputs: RunOutputs) -> dict[str, Any]:
    return {
        "evaluation_uses_all_branches": True,
        "fusion_input_enabled_mean": {
            name: float(outputs.branch_keep_masks[:, index].mean())
            for index, name in enumerate(BPUnifiedThreeBranch.BRANCH_NAMES)
        },
        "fingerprint_weight_mean": {
            name: float(outputs.fp_weights[:, index].mean())
            for index, name in enumerate(BPUnifiedThreeBranch.FP_VIEW_NAMES)
        },
        "branch_metrics": {
            name: evaluate_binary_metrics(outputs.labels, probabilities)
            for name, probabilities in outputs.branch_probs.items()
        },
    }


def branch_module_pair(model: BPUnifiedThreeBranch, branch_name: str) -> tuple[nn.Module, nn.Module]:
    mapping = {
        "graph": (model.graph_branch, model.graph_head),
        "fingerprint": (model.fp_branch, model.fp_head),
        "unimol": (model.unimol_atom_branch, model.unimol_head),
    }
    if branch_name not in mapping:
        raise ValueError(f"Unknown branch: {branch_name}")
    return mapping[branch_name]


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def freeze_entire_model(model: nn.Module) -> None:
    set_requires_grad(model, False)


def set_branch_trainable(model: BPUnifiedThreeBranch, branch_name: str, enabled: bool) -> None:
    encoder, head = branch_module_pair(model, branch_name)
    set_requires_grad(encoder, enabled)
    set_requires_grad(head, enabled)


def branch_parameters(model: BPUnifiedThreeBranch, branch_name: str) -> list[nn.Parameter]:
    encoder, head = branch_module_pair(model, branch_name)
    return list(encoder.parameters()) + list(head.parameters())


def cpu_module_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def capture_branch_state(model: BPUnifiedThreeBranch, branch_name: str) -> dict[str, Any]:
    encoder, head = branch_module_pair(model, branch_name)
    return {"encoder": cpu_module_state(encoder), "head": cpu_module_state(head)}


def restore_branch_state(model: BPUnifiedThreeBranch, branch_name: str, state: Mapping[str, Any]) -> None:
    encoder, head = branch_module_pair(model, branch_name)
    encoder.load_state_dict(state["encoder"])
    head.load_state_dict(state["head"])


def force_fully_frozen_modules_to_eval(model: nn.Module) -> None:
    for module in model.modules():
        parameters = list(module.parameters(recurse=True))
        if parameters and not any(parameter.requires_grad for parameter in parameters):
            module.eval()


def branch_metrics_from_outputs(outputs: RunOutputs) -> dict[str, dict[str, float]]:
    return {
        name: evaluate_binary_metrics(outputs.labels, probabilities)
        for name, probabilities in outputs.branch_probs.items()
    }


def load_stage_checkpoint(path: Path, signature: str, phase: str) -> dict[str, Any]:
    checkpoint = torch_load(path)
    if checkpoint.get("task_signature") != signature or checkpoint.get("phase") != phase:
        raise RuntimeError(
            f"Checkpoint configuration mismatch: {path}. "
            "Use a different --run-name for changed settings or split files."
        )
    return checkpoint


def write_history(path: Path, history: Sequence[Mapping[str, Any]]) -> None:
    atomic_write_frame(path, pd.DataFrame(history))


def train_phase1_branches(
    model: BPUnifiedThreeBranch,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
    output_dir: Path,
    signature: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    phase = "phase1"
    last_path = output_dir / "phase1_last.pt"
    complete_path = output_dir / "phase1_complete.pt"
    history_path = output_dir / "train_log_phase1_branches.csv"
    if complete_path.is_file():
        checkpoint = load_stage_checkpoint(complete_path, signature, phase)
        model.load_state_dict(checkpoint["model_state"])
        restore_rng_state(checkpoint.get("rng_state"))
        return checkpoint["history"], checkpoint["summary"]

    names = list(BPUnifiedThreeBranch.BRANCH_NAMES)
    freeze_entire_model(model)
    for name in names:
        set_branch_trainable(model, name, True)
    optimizers = {
        name: torch.optim.AdamW(
            branch_parameters(model, name),
            lr=args.branch_learning_rate,
            weight_decay=args.weight_decay,
        )
        for name in names
    }
    schedulers = {
        name: torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizers[name], mode="max", factor=0.6, patience=5, min_lr=1e-6
        )
        for name in names
    }
    ema = ModelEMA(model, args.ema_decay, args.ema_warmup_steps) if args.use_ema else None
    best_scores = {name: -np.inf for name in names}
    best_epochs = {name: 0 for name in names}
    best_metrics: dict[str, Any] = {name: None for name in names}
    best_states: dict[str, Any] = {name: None for name in names}
    wait = {name: 0 for name in names}
    active = set(names)
    history: list[dict[str, Any]] = []
    start_epoch = 1

    if last_path.is_file():
        checkpoint = load_stage_checkpoint(last_path, signature, phase)
        model.load_state_dict(checkpoint["model_state"])
        for name in names:
            optimizers[name].load_state_dict(checkpoint["optimizers"][name])
            schedulers[name].load_state_dict(checkpoint["schedulers"][name])
        if ema is not None and checkpoint.get("ema") is not None:
            ema.load_checkpoint_state(checkpoint["ema"])
        state = checkpoint["state"]
        best_scores, best_epochs = state["best_scores"], state["best_epochs"]
        best_metrics, best_states = state["best_metrics"], state["best_states"]
        wait, active, history = state["wait"], set(state["active"]), state["history"]
        start_epoch = int(checkpoint["next_epoch"])
        freeze_entire_model(model)
        for name in active:
            set_branch_trainable(model, name, True)
        restore_rng_state(checkpoint.get("rng_state"))
        print(f"[{args.dataset}][seed={seed}] Resuming phase 1 at epoch {start_epoch}")

    for epoch in range(start_epoch, args.branch_max_epochs + 1):
        if not active:
            break
        model.train()
        force_fully_frozen_modules_to_eval(model)
        loss_sums = {name: 0.0 for name in names}
        batch_count = 0
        progress = tqdm(
            train_loader,
            desc=f"{args.dataset} seed={seed} phase1 {epoch}/{args.branch_max_epochs}",
            leave=False,
        )
        for batch in progress:
            labels = batch["labels"].to(device, non_blocking=True)
            for name in active:
                optimizers[name].zero_grad(set_to_none=True)
            output = batch_forward(model, batch, device)
            losses: dict[str, torch.Tensor] = {}
            for name in names:
                logits = output[BRANCH_TO_LOGIT[name]]
                loss = criterion(logits, labels)
                if args.use_rank:
                    loss = loss + args.rank_loss_weight * pairwise_ranking_loss(
                        logits, labels, margin=args.rank_margin
                    )
                losses[name] = loss
            active_loss = torch.stack([losses[name] for name in sorted(active)]).sum()
            active_loss.backward()
            active_parameters = [
                parameter
                for name in active
                for parameter in branch_parameters(model, name)
                if parameter.requires_grad
            ]
            torch.nn.utils.clip_grad_norm_(active_parameters, args.grad_clip_norm)
            for name in active:
                optimizers[name].step()
            if ema is not None:
                ema.update(model)
            for name in names:
                loss_sums[name] += float(losses[name].detach().cpu())
            batch_count += 1

        evaluation_model = ema.module if ema is not None else model
        validation = run_eval(evaluation_model, val_loader, device)
        branch_metrics = branch_metrics_from_outputs(validation)
        row: dict[str, Any] = {
            "phase": "branch_pretraining",
            "epoch": epoch,
            "active_branches": ",".join(sorted(active)),
        }
        newly_stopped: list[str] = []
        for name in names:
            metrics = branch_metrics[name]
            score = compute_val_score(metrics)
            if name in active:
                schedulers[name].step(score if np.isfinite(score) else -np.inf)
            row[f"train_loss_{name}"] = loss_sums[name] / max(batch_count, 1)
            row[f"val_score_{name}"] = score
            row[f"val_AUPRC_{name}"] = metrics["AUPRC"]
            row[f"val_ROC-AUC_{name}"] = metrics["ROC-AUC"]
            row[f"val_MCC_{name}"] = metrics["MCC"]
            row[f"lr_{name}"] = float(optimizers[name].param_groups[0]["lr"])
            if name not in active:
                continue
            if np.isfinite(score) and score > best_scores[name]:
                best_scores[name] = float(score)
                best_epochs[name] = int(epoch)
                best_metrics[name] = dict(metrics)
                best_states[name] = capture_branch_state(evaluation_model, name)
                wait[name] = 0
            else:
                wait[name] += 1
                if wait[name] >= args.branch_patience:
                    if best_states[name] is None:
                        raise RuntimeError(f"Phase 1 branch {name} has no valid checkpoint")
                    restore_branch_state(model, name, best_states[name])
                    set_branch_trainable(model, name, False)
                    newly_stopped.append(name)
        active.difference_update(newly_stopped)
        history.append(row)
        write_history(history_path, history)
        atomic_torch_save(
            last_path,
            {
                "phase": phase,
                "task_signature": signature,
                "next_epoch": epoch + 1,
                "model_state": cpu_module_state(model),
                "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
                "schedulers": {name: scheduler.state_dict() for name, scheduler in schedulers.items()},
                "ema": ema.checkpoint_state() if ema is not None else None,
                "state": {
                    "best_scores": best_scores,
                    "best_epochs": best_epochs,
                    "best_metrics": best_metrics,
                    "best_states": best_states,
                    "wait": wait,
                    "active": sorted(active),
                    "history": history,
                },
                "rng_state": capture_rng_state(),
            },
        )
        summary_text = " | ".join(
            f"{name}:now={row[f'val_score_{name}']:.4f},best={best_scores[name]:.4f}"
            for name in names
        )
        print(f"[Phase1 {epoch:03d}] active={sorted(active)} | {summary_text}")

    for name in names:
        if best_states[name] is None:
            raise RuntimeError(f"Phase 1 branch {name} has no valid checkpoint")
        restore_branch_state(model, name, best_states[name])
        set_branch_trainable(model, name, False)
    summary = {
        name: {
            "best_epoch": int(best_epochs[name]),
            "best_val_score": float(best_scores[name]),
            "best_val_metrics": best_metrics[name],
        }
        for name in names
    }
    atomic_torch_save(
        complete_path,
        {
            "phase": phase,
            "task_signature": signature,
            "model_state": cpu_module_state(model),
            "history": history,
            "summary": summary,
            "rng_state": capture_rng_state(),
        },
    )
    return history, summary


def train_phase2_fusion(
    model: BPUnifiedThreeBranch,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
    output_dir: Path,
    signature: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    phase = "phase2"
    last_path = output_dir / "phase2_last.pt"
    complete_path = output_dir / "phase2_complete.pt"
    history_path = output_dir / "train_log_phase2_fusion.csv"
    if complete_path.is_file():
        checkpoint = load_stage_checkpoint(complete_path, signature, phase)
        model.load_state_dict(checkpoint["model_state"])
        restore_rng_state(checkpoint.get("rng_state"))
        return checkpoint["history"], checkpoint["summary"]

    freeze_entire_model(model)
    set_requires_grad(model.fusion, True)
    optimizer = torch.optim.AdamW(
        model.fusion.parameters(),
        lr=args.fusion_learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.6, patience=5, min_lr=1e-6
    )
    ema = ModelEMA(model, args.ema_decay, args.ema_warmup_steps) if args.use_ema else None
    best_state = None
    best_score = -np.inf
    best_epoch = 0
    best_metrics = None
    wait = 0
    history: list[dict[str, Any]] = []
    start_epoch = 1

    if last_path.is_file():
        checkpoint = load_stage_checkpoint(last_path, signature, phase)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if ema is not None and checkpoint.get("ema") is not None:
            ema.load_checkpoint_state(checkpoint["ema"])
        state = checkpoint["state"]
        best_state, best_score = state["best_state"], float(state["best_score"])
        best_epoch, best_metrics = int(state["best_epoch"]), state["best_metrics"]
        wait, history = int(state["wait"]), state["history"]
        start_epoch = int(checkpoint["next_epoch"])
        freeze_entire_model(model)
        set_requires_grad(model.fusion, True)
        restore_rng_state(checkpoint.get("rng_state"))
        print(f"[{args.dataset}][seed={seed}] Resuming phase 2 at epoch {start_epoch}")

    for epoch in range(start_epoch, args.fusion_max_epochs + 1):
        if wait >= args.fusion_patience:
            break
        model.train()
        force_fully_frozen_modules_to_eval(model)
        total_loss = 0.0
        keep_sum = torch.zeros(3, dtype=torch.float32, device=device)
        keep_count = batch_count = 0
        for batch in tqdm(
            train_loader,
            desc=f"{args.dataset} seed={seed} phase2 {epoch}/{args.fusion_max_epochs}",
            leave=False,
        ):
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = batch_forward(model, batch, device)
            loss = criterion(output["logit"], labels)
            if args.use_rank:
                loss = loss + args.rank_loss_weight * pairwise_ranking_loss(
                    output["logit"], labels, margin=args.rank_margin
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.fusion.parameters(), args.grad_clip_norm)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            total_loss += float(loss.detach().cpu())
            keep_sum.add_(output["branch_keep_mask"].detach().float().sum(dim=0))
            keep_count += int(output["branch_keep_mask"].size(0))
            batch_count += 1

        evaluation_model = ema.module if ema is not None else model
        validation = run_eval(evaluation_model, val_loader, device)
        metrics = evaluate_binary_metrics(validation.labels, validation.probs)
        score = compute_val_score(metrics)
        scheduler.step(score if np.isfinite(score) else -np.inf)
        keep_rates = (keep_sum / max(keep_count, 1)).cpu().tolist()
        history.append({
            "phase": "frozen_fusion",
            "epoch": epoch,
            "train_loss": total_loss / max(batch_count, 1),
            "val_score": score,
            **{f"val_{key}": value for key, value in metrics.items()},
            "keep_graph": float(keep_rates[0]),
            "keep_fingerprint": float(keep_rates[1]),
            "keep_unimol": float(keep_rates[2]),
            "lr_fusion": float(optimizer.param_groups[0]["lr"]),
        })
        if np.isfinite(score) and score > best_score:
            best_score, best_epoch = float(score), int(epoch)
            best_metrics, best_state = dict(metrics), cpu_module_state(evaluation_model)
            wait = 0
        else:
            wait += 1
        write_history(history_path, history)
        atomic_torch_save(
            last_path,
            {
                "phase": phase,
                "task_signature": signature,
                "next_epoch": epoch + 1,
                "model_state": cpu_module_state(model),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "ema": ema.checkpoint_state() if ema is not None else None,
                "state": {
                    "best_state": best_state,
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                    "best_metrics": best_metrics,
                    "wait": wait,
                    "history": history,
                },
                "rng_state": capture_rng_state(),
            },
        )
        print(
            f"[Phase2 {epoch:03d}] AUPRC={metrics['AUPRC']:.4f} | "
            f"ROC-AUC={metrics['ROC-AUC']:.4f} | MCC={metrics['MCC']:.4f} | "
            f"score={score:.4f} | best={best_score:.4f}"
        )
        if wait >= args.fusion_patience:
            break

    if best_state is None:
        raise RuntimeError("Phase 2 produced no valid fusion checkpoint")
    model.load_state_dict(best_state)
    summary = {
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_score),
        "best_val_metrics": best_metrics,
    }
    atomic_torch_save(
        complete_path,
        {
            "phase": phase,
            "task_signature": signature,
            "model_state": cpu_module_state(model),
            "history": history,
            "summary": summary,
            "rng_state": capture_rng_state(),
        },
    )
    return history, summary


def configure_top_layer_finetuning(
    model: BPUnifiedThreeBranch,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    freeze_entire_model(model)
    set_requires_grad(model.fusion, True)
    top_modules = [
        model.graph_branch.out,
        model.graph_head,
        model.fp_branch.view_score,
        model.fp_branch.concat_proj,
        model.fp_branch.norm,
        model.fp_head,
        model.unimol_atom_branch.out,
        model.unimol_head,
    ]
    for module in top_modules:
        set_requires_grad(module, True)
    branch_top_parameters: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in top_modules:
        for parameter in module.parameters():
            if id(parameter) not in seen:
                branch_top_parameters.append(parameter)
                seen.add(id(parameter))
    return list(model.fusion.parameters()), branch_top_parameters


def train_phase3_finetune(
    model: BPUnifiedThreeBranch,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
    output_dir: Path,
    signature: str,
    phase2_summary: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    phase = "phase3"
    last_path = output_dir / "phase3_last.pt"
    complete_path = output_dir / "phase3_complete.pt"
    history_path = output_dir / "train_log_phase3_finetune.csv"
    if complete_path.is_file():
        checkpoint = load_stage_checkpoint(complete_path, signature, phase)
        model.load_state_dict(checkpoint["model_state"])
        restore_rng_state(checkpoint.get("rng_state"))
        return checkpoint["history"], checkpoint["summary"]

    baseline_state = cpu_module_state(model)
    baseline_score = float(phase2_summary["best_val_score"])
    if args.finetune_max_epochs <= 0:
        summary = {
            "enabled": False,
            "accepted": False,
            "best_epoch": 0,
            "best_val_score": baseline_score,
            "best_val_metrics": phase2_summary["best_val_metrics"],
        }
        atomic_torch_save(
            complete_path,
            {
                "phase": phase,
                "task_signature": signature,
                "model_state": baseline_state,
                "history": [],
                "summary": summary,
                "rng_state": capture_rng_state(),
            },
        )
        return [], summary

    fusion_parameters, branch_top_parameters = configure_top_layer_finetuning(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": fusion_parameters, "lr": args.finetune_fusion_learning_rate},
            {"params": branch_top_parameters, "lr": args.finetune_branch_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.6,
        patience=3,
        min_lr=args.finetune_branch_learning_rate / 5.0,
    )
    ema = ModelEMA(model, args.ema_decay, args.ema_warmup_steps) if args.use_ema else None
    best_state = baseline_state
    best_score = baseline_score
    best_epoch = 0
    best_metrics = phase2_summary["best_val_metrics"]
    accepted = False
    wait = 0
    history: list[dict[str, Any]] = []
    start_epoch = 1

    if last_path.is_file():
        checkpoint = load_stage_checkpoint(last_path, signature, phase)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if ema is not None and checkpoint.get("ema") is not None:
            ema.load_checkpoint_state(checkpoint["ema"])
        state = checkpoint["state"]
        baseline_score = float(state["baseline_score"])
        best_state, best_score = state["best_state"], float(state["best_score"])
        best_epoch, best_metrics = int(state["best_epoch"]), state["best_metrics"]
        accepted, wait = bool(state["accepted"]), int(state["wait"])
        history, start_epoch = state["history"], int(checkpoint["next_epoch"])
        configure_top_layer_finetuning(model)
        restore_rng_state(checkpoint.get("rng_state"))
        print(f"[{args.dataset}][seed={seed}] Resuming phase 3 at epoch {start_epoch}")

    for epoch in range(start_epoch, args.finetune_max_epochs + 1):
        if wait >= args.finetune_patience:
            break
        model.train()
        force_fully_frozen_modules_to_eval(model)
        total_loss = 0.0
        batch_count = 0
        for batch in tqdm(
            train_loader,
            desc=f"{args.dataset} seed={seed} phase3 {epoch}/{args.finetune_max_epochs}",
            leave=False,
        ):
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = batch_forward(model, batch, device)
            main_loss = criterion(output["logit"], labels)
            auxiliary_loss = sum(
                criterion(output[BRANCH_TO_LOGIT[name]], labels)
                for name in BPUnifiedThreeBranch.BRANCH_NAMES
            )
            loss = main_loss + args.finetune_aux_weight * auxiliary_loss
            if args.use_rank:
                loss = loss + args.rank_loss_weight * pairwise_ranking_loss(
                    output["logit"], labels, margin=args.rank_margin
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                fusion_parameters + branch_top_parameters, args.grad_clip_norm
            )
            optimizer.step()
            if ema is not None:
                ema.update(model)
            total_loss += float(loss.detach().cpu())
            batch_count += 1

        evaluation_model = ema.module if ema is not None else model
        validation = run_eval(evaluation_model, val_loader, device)
        metrics = evaluate_binary_metrics(validation.labels, validation.probs)
        score = compute_val_score(metrics)
        scheduler.step(score if np.isfinite(score) else -np.inf)
        history.append({
            "phase": "protected_top_finetuning",
            "epoch": epoch,
            "train_loss": total_loss / max(batch_count, 1),
            "val_score": score,
            **{f"val_{key}": value for key, value in metrics.items()},
            "lr_fusion": float(optimizer.param_groups[0]["lr"]),
            "lr_branch_top": float(optimizer.param_groups[1]["lr"]),
            "improves_phase2": bool(np.isfinite(score) and score > baseline_score),
        })
        if np.isfinite(score) and score > best_score:
            best_score, best_epoch = float(score), int(epoch)
            best_metrics, best_state = dict(metrics), cpu_module_state(evaluation_model)
            accepted, wait = True, 0
        else:
            wait += 1
        write_history(history_path, history)
        atomic_torch_save(
            last_path,
            {
                "phase": phase,
                "task_signature": signature,
                "next_epoch": epoch + 1,
                "model_state": cpu_module_state(model),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "ema": ema.checkpoint_state() if ema is not None else None,
                "state": {
                    "baseline_score": baseline_score,
                    "best_state": best_state,
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                    "best_metrics": best_metrics,
                    "accepted": accepted,
                    "wait": wait,
                    "history": history,
                },
                "rng_state": capture_rng_state(),
            },
        )
        print(
            f"[Phase3 {epoch:03d}] score={score:.4f} | phase2={baseline_score:.4f} | "
            f"best={best_score:.4f} | accepted={accepted}"
        )
        if wait >= args.finetune_patience:
            break

    model.load_state_dict(best_state)
    summary = {
        "enabled": True,
        "accepted": bool(accepted),
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_score),
        "best_val_metrics": best_metrics,
    }
    atomic_torch_save(
        complete_path,
        {
            "phase": phase,
            "task_signature": signature,
            "model_state": cpu_module_state(model),
            "history": history,
            "summary": summary,
            "rng_state": capture_rng_state(),
        },
    )
    return history, summary


def task_config(runtime: DatasetRuntime, seed: int) -> dict[str, Any]:
    args = runtime.args
    paths = runtime.split_paths[seed]
    relevant_ids = sorted(
        {
            entity_id_for_smiles(smiles)
            for frame in runtime.splits[seed].values()
            for smiles in frame["SMILES"].astype(str)
        }
    )
    return {
        "dataset": runtime.dataset,
        "seed": seed,
        "scenario": SCENARIO,
        "variant": VARIANT_NAME,
        "architecture_version": ARCHITECTURE_VERSION,
        "architecture": {
            "model_dim": MODEL_DIM,
            "graph_layers": GRAPH_LAYERS,
            "readout_steps": READOUT_STEPS,
            "dropout": DROPOUT,
            "branch_dropout": BRANCH_DROPOUT,
            "unimol_atom_dim": UNIMOL_ATOM_DIM,
            "atom_feature_dim": ATOM_FEAT_DIM,
            "bond_feature_dim": BOND_FEAT_DIM,
            "fingerprint_dims": FP_DIMS,
            "variant_config": VARIANT_CONFIG,
        },
        "split_sha256": {name: hash_file(path) for name, path in paths.items()},
        "feature_manifest_fingerprint": {
            "unimol2": runtime.unimol_index.fingerprint(relevant_ids),
            "rdkit": runtime.rdkit_index.fingerprint(relevant_ids),
        },
        "training": {
            "branch_max_epochs": args.branch_max_epochs,
            "branch_patience": args.branch_patience,
            "fusion_max_epochs": args.fusion_max_epochs,
            "fusion_patience": args.fusion_patience,
            "finetune_max_epochs": args.finetune_max_epochs,
            "finetune_patience": args.finetune_patience,
            "train_batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "branch_learning_rate": args.branch_learning_rate,
            "fusion_learning_rate": args.fusion_learning_rate,
            "finetune_fusion_learning_rate": args.finetune_fusion_learning_rate,
            "finetune_branch_learning_rate": args.finetune_branch_learning_rate,
            "weight_decay": args.weight_decay,
            "grad_clip_norm": args.grad_clip_norm,
            "finetune_aux_weight": args.finetune_aux_weight,
            "use_rank": args.use_rank,
            "rank_loss_weight": args.rank_loss_weight,
            "rank_margin": args.rank_margin,
            "use_ema": args.use_ema,
            "ema_decay": args.ema_decay,
            "ema_warmup_steps": args.ema_warmup_steps,
            "deterministic": args.deterministic,
        },
        "checkpoint_selection": {
            "threshold": THRESHOLD,
            "weights": VAL_SCORE_WEIGHTS,
        },
    }


def run_task(runtime: DatasetRuntime, store: FeatureStore, seed: int) -> dict[str, Any]:
    args = runtime.args
    output_dir = args.dataset_model_root / f"seed_{seed:04d}"
    status_path = output_dir / "status.json"
    metrics_path = output_dir / "all_metrics.json"
    config = task_config(runtime, seed)
    signature = hash_mapping(config)
    config["task_signature"] = signature
    status = load_json(status_path, {}) or {}
    existing = load_json(metrics_path, None)
    if status.get("completed") and status.get("task_signature") == signature and existing:
        print(f"[{runtime.dataset}][seed={seed}] Task already complete; skipping.")
        return existing

    frames = runtime.splits[seed]
    train_set = BBBUnifiedDataset(frames["train"], store)
    val_set = BBBUnifiedDataset(frames["val"], store)
    test_set = BBBUnifiedDataset(frames["test"], store)
    train_loader = make_data_loader(train_set, args.batch_size, True, args)
    val_loader = make_data_loader(val_set, args.eval_batch_size, False, args)
    test_loader = make_data_loader(test_set, args.eval_batch_size, False, args)

    seed_everything(seed, args.deterministic)
    device = torch.device(args.device)
    model = BPUnifiedThreeBranch(
        UNIMOL_ATOM_DIM, ATOM_FEAT_DIM, BOND_FEAT_DIM, VARIANT_CONFIG
    ).to(device)
    total_parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    labels = frames["train"]["label"].to_numpy(dtype=int)
    positive = max(int((labels == 1).sum()), 1)
    negative = max(int((labels == 0).sum()), 1)
    pos_weight = torch.tensor([negative / positive], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    started = time.time()

    _, phase1_summary = train_phase1_branches(
        model, train_loader, val_loader, criterion, device, args, seed,
        output_dir, signature,
    )
    _, phase2_summary = train_phase2_fusion(
        model, train_loader, val_loader, criterion, device, args, seed,
        output_dir, signature,
    )
    _, phase3_summary = train_phase3_finetune(
        model, train_loader, val_loader, criterion, device, args, seed,
        output_dir, signature, phase2_summary,
    )
    selected_phase = 3 if phase3_summary["accepted"] else 2
    selected_summary = phase3_summary if selected_phase == 3 else phase2_summary
    freeze_entire_model(model)

    validation = run_eval(model, val_loader, device)
    test = run_eval(model, test_loader, device)
    validation_metrics = evaluate_binary_metrics(validation.labels, validation.probs)
    test_metrics = evaluate_binary_metrics(test.labels, test.probs)
    training_seconds = float(time.time() - started)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_frame(output_dir / "predictions_val.csv", prediction_frame(validation))
    atomic_write_frame(output_dir / "predictions_test.csv", prediction_frame(test))
    atomic_write_json(output_dir / "metrics_val.json", validation_metrics)
    atomic_write_json(output_dir / "metrics_test.json", test_metrics)
    atomic_write_json(output_dir / "diagnostics_val.json", summarize_diagnostics(validation))
    atomic_write_json(output_dir / "diagnostics_test.json", summarize_diagnostics(test))

    config.update({
        "dataset_defaults": asdict(DATASET_DEFAULTS[runtime.dataset]),
        "split_sizes": {name: len(frame) for name, frame in frames.items()},
        "positive_class_weight": float(pos_weight.item()),
        "trainable_parameters": total_parameters,
        "selected_phase": selected_phase,
        "phase1_summary": phase1_summary,
        "phase2_summary": phase2_summary,
        "phase3_summary": phase3_summary,
    })
    atomic_write_json(output_dir / "config.json", config)
    atomic_torch_save(
        output_dir / "best_model.pt",
        {
            "model_state_dict": cpu_module_state(model),
            "task_signature": signature,
            "architecture_version": ARCHITECTURE_VERSION,
            "variant_name": VARIANT_NAME,
            "variant_config": VARIANT_CONFIG,
            "selected_phase": selected_phase,
            "best_epoch": int(selected_summary["best_epoch"]),
            "best_val_score": float(selected_summary["best_val_score"]),
            "unimol_atom_dim": UNIMOL_ATOM_DIM,
            "atom_dim": ATOM_FEAT_DIM,
            "edge_dim": BOND_FEAT_DIM,
            "fingerprint_dim": FP_TOTAL_DIM,
        },
    )
    result = {
        "dataset": runtime.dataset,
        "seed": seed,
        "variant": VARIANT_NAME,
        "selected_phase": selected_phase,
        "best_epoch": int(selected_summary["best_epoch"]),
        "best_val_score": float(selected_summary["best_val_score"]),
        "trainable_parameters": total_parameters,
        "training_seconds": training_seconds,
        "validation": validation_metrics,
        "test": test_metrics,
    }
    atomic_write_json(metrics_path, result)
    atomic_write_json(
        status_path,
        {
            "completed": True,
            "task_signature": signature,
            "finished_at": time.time(),
        },
    )
    print(
        f"[{runtime.dataset}][seed={seed}] test ROC-AUC={test_metrics['ROC-AUC']:.4f}, "
        f"AUPRC={test_metrics['AUPRC']:.4f}, MCC={test_metrics['MCC']:.4f}"
    )
    return result


METRIC_NAMES = (
    "ROC-AUC", "AUPRC", "Accuracy", "F1-score", "Precision",
    "Recall", "MCC", "Specificity",
)


def aggregate_results(runtime: DatasetRuntime, results: Sequence[Mapping[str, Any]]) -> None:
    if not results:
        return
    detail_rows = []
    for result in results:
        row = {
            "dataset": result["dataset"],
            "seed": result["seed"],
            "selected_phase": result["selected_phase"],
            "best_epoch": result["best_epoch"],
            "best_val_score": result["best_val_score"],
            "training_seconds": result["training_seconds"],
            "trainable_parameters": result["trainable_parameters"],
        }
        row.update({name: result["test"].get(name, np.nan) for name in METRIC_NAMES})
        detail_rows.append(row)
    detail = pd.DataFrame(detail_rows).sort_values("seed")
    summary: dict[str, Any] = {
        "dataset": runtime.dataset,
        "n_seeds": int(len(detail)),
        "phase2_selected_count": int((detail["selected_phase"] == 2).sum()),
        "phase3_selected_count": int((detail["selected_phase"] == 3).sum()),
        "best_epoch_mean": float(detail["best_epoch"].mean()),
        "training_seconds_mean": float(detail["training_seconds"].mean()),
    }
    for metric in METRIC_NAMES:
        summary[f"{metric}_mean"] = float(detail[metric].mean())
        summary[f"{metric}_std"] = float(detail[metric].std(ddof=0))
    root = runtime.args.dataset_model_root
    atomic_write_frame(root / "all_seed_test_metrics.csv", detail)
    atomic_write_frame(root / "summary_mean_std.csv", pd.DataFrame([summary]))

    merged = None
    used_seeds: list[int] = []
    for seed in sorted(int(value) for value in detail["seed"]):
        path = root / f"seed_{seed:04d}" / "predictions_test.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path)[["SMILES", "y_true", "y_pred_proba"]].copy()
        frame = frame.rename(columns={"y_pred_proba": f"prob_seed_{seed}"})
        merged = frame if merged is None else merged.merge(
            frame, on=["SMILES", "y_true"], how="inner", validate="one_to_one"
        )
        used_seeds.append(seed)
    if merged is not None and used_seeds:
        columns = [f"prob_seed_{seed}" for seed in used_seeds]
        merged["y_pred_proba_ensemble"] = merged[columns].mean(axis=1)
        merged["y_pred_class_ensemble"] = (
            merged["y_pred_proba_ensemble"] >= THRESHOLD
        ).astype(int)
        metrics = evaluate_binary_metrics(
            merged["y_true"].to_numpy(), merged["y_pred_proba_ensemble"].to_numpy()
        )
        atomic_write_frame(root / "seed_ensemble_predictions_test.csv", merged)
        atomic_write_json(
            root / "seed_ensemble_metrics.json",
            {"n_models": len(used_seeds), "seeds": used_seeds, **metrics},
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the BP-NET staged unified three-branch model on B3DB and BBBP."
    )
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--seeds", default="1-5")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--embedding-root", type=Path, default=DEFAULT_EMBEDDING_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--non-deterministic", dest="deterministic", action="store_false")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true", default=True)
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--preload-to-ram", action="store_true", default=True)
    parser.add_argument("--no-preload-to-ram", dest="preload_to_ram", action="store_false")
    parser.add_argument("--feature-cache-size", type=int, default=4096)

    parser.add_argument("--branch-max-epochs", type=int, default=None)
    parser.add_argument("--branch-patience", type=int, default=None)
    parser.add_argument("--fusion-max-epochs", type=int, default=None)
    parser.add_argument("--fusion-patience", type=int, default=None)
    parser.add_argument("--finetune-max-epochs", type=int, default=None)
    parser.add_argument("--finetune-patience", type=int, default=None)
    parser.add_argument("--branch-learning-rate", type=float, default=1e-4)
    parser.add_argument("--fusion-learning-rate", type=float, default=1e-4)
    parser.add_argument("--finetune-fusion-learning-rate", type=float, default=2e-5)
    parser.add_argument("--finetune-branch-learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--finetune-aux-weight", type=float, default=0.05)
    parser.add_argument("--use-rank", action="store_true", default=True)
    parser.add_argument("--no-rank", dest="use_rank", action="store_false")
    parser.add_argument("--rank-loss-weight", type=float, default=0.02)
    parser.add_argument("--rank-margin", type=float, default=0.2)
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--no-ema", dest="use_ema", action="store_false")
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--ema-warmup-steps", type=int, default=100)
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    run_name = str(args.run_name).strip()
    if not run_name or run_name in {".", ".."} or Path(run_name).name != run_name:
        raise ValueError("--run-name must be one directory name")
    positive = {
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "prefetch_factor": args.prefetch_factor,
        "feature_cache_size": args.feature_cache_size,
        "ema_warmup_steps": args.ema_warmup_steps,
    }
    for name in (
        "branch_max_epochs", "branch_patience", "fusion_max_epochs",
        "fusion_patience", "finetune_patience",
    ):
        value = getattr(args, name)
        if value is not None:
            positive[name] = value
    bad = [name for name, value in positive.items() if value <= 0]
    if bad:
        raise ValueError(f"These arguments must be positive: {bad}")
    if args.finetune_max_epochs is not None and args.finetune_max_epochs < 0:
        raise ValueError("--finetune-max-epochs must be >= 0")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    rates = (
        args.branch_learning_rate, args.fusion_learning_rate,
        args.finetune_fusion_learning_rate, args.finetune_branch_learning_rate,
    )
    if any(value <= 0 for value in rates):
        raise ValueError("Learning rates must be positive")
    if args.weight_decay < 0 or args.grad_clip_norm <= 0:
        raise ValueError("Weight decay/gradient clipping values are invalid")
    if not 0 <= args.ema_decay < 1:
        raise ValueError("--ema-decay must be in [0, 1)")
    if args.rank_loss_weight < 0 or args.rank_margin < 0 or args.finetune_aux_weight < 0:
        raise ValueError("Loss weights/margin cannot be negative")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    base = parser.parse_args(argv)
    validate_arguments(base)
    base.data_root = Path(base.data_root)
    base.embedding_root = Path(base.embedding_root)
    base.model_root = Path(base.model_root)
    base.datasets = normalize_datasets(base.datasets)
    base.seeds = parse_seeds(base.seeds)
    base.device = resolve_device(base.device)

    runtimes: list[DatasetRuntime] = []
    for dataset in base.datasets:
        dataset_args = resolve_dataset_args(base, dataset)
        validate_arguments(dataset_args)
        runtimes.append(preflight_dataset(dataset, dataset_args))
    if base.check_only:
        print("Preflight complete. No model files were written.")
        return 0

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    for runtime in runtimes:
        args = runtime.args
        store = FeatureStore(
            runtime.unimol_index, runtime.rdkit_index, args.feature_cache_size
        )
        if args.preload_to_ram:
            store.preload(
                runtime.entity_ids,
                f"{runtime.dataset} BP-NET features",
            )
        else:
            # Fail on dimensional incompatibility before creating the first task.
            store.get(runtime.entity_ids[0])
        results: list[dict[str, Any]] = []
        for seed in args.seeds:
            results.append(run_task(runtime, store, seed))
            # Preserve aggregate progress even if a later seed is interrupted.
            aggregate_results(runtime, results)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
