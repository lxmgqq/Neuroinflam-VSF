#!/usr/bin/env python3
"""
Train UE-AlignNet on Davis, KIBA, and BindingDB.

The script consumes precomputed dataset splits and feature artifacts generated
by the Neuroinflam-VSF preprocessing pipeline.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

def _bootstrap_conda_libstdcxx():
    """Prefer the active conda libstdc++ before importing PyTorch and restart once if needed."""
    if os.environ.get("UEALIGNNET_LIBSTDCXX_BOOTSTRAPPED") == "1":
        return

    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if not conda_prefix:
        exe = Path(sys.executable).resolve()
        if exe.parent.name == "bin":
            conda_prefix = str(exe.parent.parent)

    if not conda_prefix:
        return

    conda_lib = os.path.join(conda_prefix, "lib")
    conda_libstdcpp = os.path.join(conda_lib, "libstdc++.so.6")
    if not os.path.exists(conda_libstdcpp):
        return

    old_ld = os.environ.get("LD_LIBRARY_PATH", "")
    old_parts = [x for x in old_ld.split(":") if x]
    if old_parts and os.path.abspath(old_parts[0]) == os.path.abspath(conda_lib):
        os.environ["UEALIGNNET_LIBSTDCXX_BOOTSTRAPPED"] = "1"
        return

    new_parts = [conda_lib] + [x for x in old_parts if os.path.abspath(x) != os.path.abspath(conda_lib)]
    new_env = os.environ.copy()
    new_env["LD_LIBRARY_PATH"] = ":".join(new_parts)
    new_env["UEALIGNNET_LIBSTDCXX_BOOTSTRAPPED"] = "1"

    print(f"Using conda libstdc++ from {conda_lib}; restarting the script once...", flush=True)
    os.execvpe(sys.executable, [sys.executable] + sys.argv, new_env)

_bootstrap_conda_libstdcxx()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) >= 3 else Path.cwd()
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "datasets"
DEFAULT_EMBEDDING_ROOT = REPO_ROOT / "data" / "embedding"
DEFAULT_MODEL_BASE = REPO_ROOT / "models"

DEFAULT_MODEL_NAME = "UE_AlignNet"

SCENARIO_ORDER = ("warm", "cold_drug", "cold_target", "double_cold")
METRIC_ORDER = ("MSE", "CI", "Rm2", "RMSE", "MAE", "Pearson", "Spearman", "R2")
DEFAULT_SEEDS = list(range(1, 6))
DEFAULT_TARGET_SCENARIOS = ["warm", "cold_drug", "cold_target", "double_cold"]
ALLOWED_SCENARIOS = set(DEFAULT_TARGET_SCENARIOS)

DEFAULT_LIGAND_DIM = 1536
DEFAULT_PROTEIN_DIM = 1152
DEFAULT_HIDDEN_DIM = 512
DEFAULT_ATOM_FEAT_DIM = 42

DATASET_RUNTIME_DEFAULTS = {
    "Davis": {
        "batch_size": 32,
        "max_protein_len": 1500,
        "preload_all_data_to_memory": True,
        "max_epochs": 300,
        "patience": 30,
        "cold_target_max_epochs": 300,
        "cold_target_patience": 24,
        "double_cold_max_epochs": 300,
        "double_cold_patience": 30,
    },
    "KIBA": {
        "batch_size": 32,
        "max_protein_len": 1500,
        "preload_all_data_to_memory": True,
        "max_epochs": 300,
        "patience": 30,
        "cold_target_max_epochs": 300,
        "cold_target_patience": 24,
        "double_cold_max_epochs": 300,
        "double_cold_patience": 30,
    },
    "BindingDB": {
        "batch_size": 16,
        "max_protein_len": 0,
        "preload_all_data_to_memory": False,
        "max_epochs": 300,
        "patience": 20,
        "cold_target_max_epochs": 300,
        "cold_target_patience": 24,
        "double_cold_max_epochs": 300,
        "double_cold_patience": 24,
    },
}

def rm2_score(y_true, y_pred):
    """Return the DeepDTA-style r_m^2 external-prediction metric."""
    observed = np.asarray(y_true, dtype=np.float64).reshape(-1)
    predicted = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[valid]
    predicted = predicted[valid]
    if observed.size < 2:
        return float("nan")

    observed_centered = observed - observed.mean()
    predicted_centered = predicted - predicted.mean()
    denominator = np.sum(observed_centered ** 2) * np.sum(predicted_centered ** 2)
    predicted_norm = np.sum(predicted ** 2)
    observed_variance = np.sum(observed_centered ** 2)
    if denominator <= 0.0 or predicted_norm <= 0.0 or observed_variance <= 0.0:
        return float("nan")

    r2 = float(np.sum(observed_centered * predicted_centered) ** 2 / denominator)
    slope = float(np.sum(observed * predicted) / predicted_norm)
    r02 = float(1.0 - np.sum((observed - slope * predicted) ** 2) / observed_variance)
    return float(r2 * (1.0 - np.sqrt(abs(r2 * r2 - r02 * r02))))

def build_standard_summary(
    detail,
    *,
    scene_column="scene",
    dataset_column=None,
    aliases=None,
):
    """Aggregate seed results using a fixed scenario and metric order."""
    if detail.empty:
        return pd.DataFrame()

    aliases = dict(aliases or {})
    work = detail.copy()
    selected = []
    for metric in METRIC_ORDER:
        candidates = tuple(
            aliases.get(metric, (metric, f"test_{metric}", metric.lower(), f"test_{metric.lower()}"))
        )
        source = next((name for name in candidates if name in work.columns), None)
        if source is not None:
            work[metric] = pd.to_numeric(work[source], errors="coerce")
            selected.append(metric)
    if not selected:
        return pd.DataFrame()

    summary = work.groupby(scene_column, sort=False)[selected].agg(["mean", "std"])
    summary = summary.reindex([scene for scene in SCENARIO_ORDER if scene in summary.index])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index().rename(columns={scene_column: "scene"})

    ordered_columns = ["scene"]
    for metric in METRIC_ORDER:
        ordered_columns.extend(
            [name for name in (f"{metric}_mean", f"{metric}_std") if name in summary.columns]
        )
    summary = summary[ordered_columns]

    if dataset_column and dataset_column in work.columns:
        values = work[dataset_column].dropna().astype(str).unique().tolist()
        if len(values) == 1:
            summary.insert(0, "dataset", values[0])
    numeric_columns = summary.select_dtypes(include=[np.number]).columns
    summary[numeric_columns] = summary[numeric_columns].round(4)
    return summary

def format_standard_summary(summary):
    """Format a summary table with four decimal places."""
    if summary.empty:
        return "No complete results were found."
    return summary.to_string(index=False, float_format=lambda value: f"{value:.4f}")

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def save_json(obj: dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)

def load_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def md5_files(paths: Sequence[str]) -> str:
    h = hashlib.md5()
    for path in paths:
        h.update(str(path).encode("utf-8"))
        h.update(b"\0")
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        h.update(b"\0")
    return h.hexdigest()

def parse_seed_list(seed_text: Optional[str]) -> List[int]:
    if not seed_text:
        return DEFAULT_SEEDS
    seed_text = seed_text.strip()
    if seed_text.lower() == "all":
        return DEFAULT_SEEDS
    values = []
    for part in seed_text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            values.extend(list(range(int(start), int(end) + 1)))
        else:
            values.append(int(part))
    return sorted(set(values))

def parse_scenario_list(text: Optional[str]) -> List[str]:
    if not text:
        return list(DEFAULT_TARGET_SCENARIOS)
    parts = [x.strip() for x in text.split(",") if x.strip()]
    if not parts:
        return list(DEFAULT_TARGET_SCENARIOS)
    bad = [x for x in parts if x not in ALLOWED_SCENARIOS]
    if bad:
        raise ValueError(f"Invalid scenario names: {bad}. Allowed values: {sorted(ALLOWED_SCENARIOS)}")
    seen = []
    for p in parts:
        if p not in seen:
            seen.append(p)
    return seen

def set_global_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass

def ensure_scene_splits_exist(args) -> None:
    ensure_dir(args.splits_dir)
    missing = []
    for seed in args.seeds:
        seed_dir = os.path.join(args.splits_dir, f"seed_{seed:04d}")
        for scene in args.target_scenarios:
            scene_dir = os.path.join(seed_dir, scene)
            for fn in ["train.csv", "val.csv", "test.csv"]:
                fp = os.path.join(scene_dir, fn)
                if not os.path.exists(fp):
                    missing.append(fp)
    if missing:
        preview = missing[:12]
        raise FileNotFoundError(
            "Selected scene split files are incomplete.\n"
            f"Missing examples: {preview}\n"
            "Generate the required split files before training."
        )

def chunked(seq: Sequence, n: int) -> Iterable[Sequence]:
    for i in range(0, len(seq), n):
        yield seq[i: i + n]

def make_downsample_indices(length: int, max_len: Optional[int]) -> np.ndarray:
    length = int(length)
    if max_len is None or max_len <= 0 or length <= max_len:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, num=int(max_len), dtype=np.int64)

def normalize_dense_adjacency(adj: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    adj = np.asarray(adj, dtype=np.float32)
    if adj.ndim != 2:
        raise ValueError(f"Expected 2D adjacency, got shape {adj.shape}")
    if adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Adjacency must be square, got shape {adj.shape}")
    adj = np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    adj = np.maximum(adj, 0.0)
    if adj.shape[0] == 0:
        return np.eye(1, dtype=np.float32)
    np.fill_diagonal(adj, np.maximum(np.diag(adj), 1.0))
    deg = adj.sum(axis=-1, keepdims=True)
    deg = np.maximum(deg, eps)
    return (adj / np.sqrt(deg * deg.T)).astype(np.float32, copy=False)

def build_band_protein_adjacency(num_nodes: int, band_width: int = 2) -> np.ndarray:
    num_nodes = int(max(1, num_nodes))
    band_width = int(max(1, band_width))
    adj = np.eye(num_nodes, dtype=np.float32)
    for offset in range(1, band_width + 1):
        weight = 1.0 / float(offset + 1)
        i = np.arange(0, num_nodes - offset, dtype=np.int64)
        j = i + offset
        adj[i, j] = np.maximum(adj[i, j], weight)
        adj[j, i] = np.maximum(adj[j, i], weight)
    return normalize_dense_adjacency(adj)

def dense_from_edge_index(
    edge_index: np.ndarray,
    edge_weight: Optional[np.ndarray] = None,
    num_nodes: Optional[int] = None,
) -> np.ndarray:
    edge_index = np.asarray(edge_index)
    if edge_index.ndim != 2:
        raise ValueError(f"edge_index must be 2D, got shape {edge_index.shape}")
    if edge_index.shape[0] == 2:
        src, dst = edge_index[0], edge_index[1]
    elif edge_index.shape[1] == 2:
        src, dst = edge_index[:, 0], edge_index[:, 1]
    else:
        raise ValueError(f"Unsupported edge_index shape: {edge_index.shape}")
    src = src.astype(np.int64, copy=False)
    dst = dst.astype(np.int64, copy=False)
    inferred_nodes = int(max(src.max(initial=0), dst.max(initial=0)) + 1) if src.size > 0 else 1
    n = int(max(num_nodes or 0, inferred_nodes, 1))
    adj = np.eye(n, dtype=np.float32)
    if edge_weight is None:
        edge_weight = np.ones_like(src, dtype=np.float32)
    edge_weight = np.asarray(edge_weight, dtype=np.float32).reshape(-1)
    m = min(len(src), len(edge_weight))
    src = src[:m]
    dst = dst[:m]
    edge_weight = edge_weight[:m]
    valid = (src >= 0) & (src < n) & (dst >= 0) & (dst < n)
    src = src[valid]
    dst = dst[valid]
    edge_weight = edge_weight[valid]
    adj[src, dst] = np.maximum(adj[src, dst], edge_weight)
    adj[dst, src] = np.maximum(adj[dst, src], edge_weight)
    return normalize_dense_adjacency(adj)

def ensure_adj_size(adj: np.ndarray, num_nodes: int) -> np.ndarray:
    num_nodes = int(max(1, num_nodes))
    adj = np.asarray(adj, dtype=np.float32)
    if adj.ndim != 2:
        raise ValueError(f"Adjacency must be 2D, got shape {adj.shape}")
    if adj.shape[0] == num_nodes and adj.shape[1] == num_nodes:
        return normalize_dense_adjacency(adj)
    out = np.eye(num_nodes, dtype=np.float32)
    n = min(num_nodes, adj.shape[0], adj.shape[1])
    out[:n, :n] = np.maximum(out[:n, :n], adj[:n, :n])
    return normalize_dense_adjacency(out)

def normalize_entity_id(x) -> str:
    """Normalize drug_id and target_id values to stable strings."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        try:
            s = str(int(float(s)))
        except Exception:
            s = s[:-2]
    return s

def torch_load_compat(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class ArtifactIndex:
    """Read one embedding_generation.py manifest without generating anything."""

    def __init__(
        self,
        embedding_root: Path,
        dataset: str,
        entity_type: str,
        feature: str,
    ):
        self.embedding_root = Path(embedding_root).resolve()
        self.dataset = str(dataset)
        self.entity_type = str(entity_type)
        self.feature = str(feature)
        self.manifest_path = (
            self.embedding_root / self.dataset / self.entity_type / self.feature / "manifest.csv"
        )
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing feature manifest: {self.manifest_path}\n"
                "Please run codes/preprocessing/embedding_generation.py first."
            )

        frame = pd.read_csv(self.manifest_path, dtype=str, keep_default_na=False)
        required = {"entity_id", "artifact", "status", "feature"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"{self.manifest_path} is missing required columns: {sorted(missing)}"
            )
        frame = frame[frame["feature"].astype(str) == self.feature].copy()
        if frame.empty:
            raise ValueError(f"No feature={self.feature!r} rows in {self.manifest_path}")

        frame["entity_id"] = frame["entity_id"].map(normalize_entity_id)
        if (frame["entity_id"] == "").any():
            raise ValueError(f"Blank entity_id in {self.manifest_path}")
        if frame["entity_id"].duplicated().any():
            dup = frame.loc[frame["entity_id"].duplicated(), "entity_id"].tolist()
            raise ValueError(f"Duplicate entity IDs in {self.manifest_path}: {dup[:10]}")

        self.paths: Dict[str, Path] = {}
        self.bad_status: Dict[str, str] = {}
        for row in frame.to_dict("records"):
            entity_id = normalize_entity_id(row["entity_id"])
            status = str(row["status"])
            if status != "valid":
                self.bad_status[entity_id] = status
                continue
            artifact_text = str(row["artifact"]).strip()
            if not artifact_text:
                raise ValueError(f"Blank artifact path for {entity_id} in {self.manifest_path}")
            path = Path(artifact_text)
            if not path.is_absolute():
                path = (self.embedding_root / path).resolve()
            else:
                path = path.resolve()
            try:
                path.relative_to(self.embedding_root)
            except ValueError as exc:
                raise ValueError(
                    f"Manifest artifact escapes embedding root: {path}"
                ) from exc
            self.paths[entity_id] = path

    def require(self, entity_ids: Iterable[str]) -> None:
        ids = list(dict.fromkeys(normalize_entity_id(x) for x in entity_ids))
        missing_rows = [x for x in ids if x not in self.paths]
        missing_files = [x for x in ids if x in self.paths and not self.paths[x].is_file()]
        if missing_rows or missing_files:
            raise FileNotFoundError(
                f"Incomplete {self.feature} artifacts in {self.manifest_path}: "
                f"missing/invalid manifest rows={len(missing_rows)} examples={missing_rows[:10]}; "
                f"missing files={len(missing_files)} examples={missing_files[:10]}.\n"
                "This training script does not generate missing features. "
                "Please rerun embedding_generation.py."
            )

    def path_for(self, entity_id: str) -> Path:
        key = normalize_entity_id(entity_id)
        if key not in self.paths:
            status = self.bad_status.get(key, "missing")
            raise KeyError(
                f"No valid {self.feature} artifact for {self.entity_type} {key}; status={status}"
            )
        return self.paths[key]


def load_ligand_features_from_manifests(
    ligand_ids: Sequence[str],
    unimol_index: ArtifactIndex,
    rdkit_index: ArtifactIndex,
    expected_dim: int,
    expected_atom_dim: int,
    max_atoms: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
    """Load precomputed UniMol2 CLS and RDKit UE graph artifacts into memory."""
    unique_ids = list(dict.fromkeys(normalize_entity_id(x) for x in ligand_ids))
    unimol_index.require(unique_ids)
    rdkit_index.require(unique_ids)

    ligand_embeddings: Dict[str, np.ndarray] = {}
    ligand_graphs: Dict[str, Dict[str, np.ndarray]] = {}
    for ligand_id in tqdm(unique_ids, desc="[Ligand] Loading precomputed UniMol2 + RDKit artifacts"):
        unimol_obj = torch_load_compat(unimol_index.path_for(ligand_id))
        if not isinstance(unimol_obj, dict) or "cls_repr" not in unimol_obj:
            raise ValueError(f"Invalid UniMol2 artifact for ligand {ligand_id}")
        cls = unimol_obj["cls_repr"]
        if not isinstance(cls, torch.Tensor):
            cls = torch.as_tensor(cls)
        cls = cls.detach().cpu().float().reshape(-1)
        if cls.numel() != int(expected_dim) or not torch.isfinite(cls).all():
            raise ValueError(
                f"Ligand {ligand_id} UniMol2 CLS mismatch: shape={tuple(cls.shape)}, "
                f"expected=({expected_dim},)"
            )
        ligand_embeddings[ligand_id] = np.ascontiguousarray(
            cls.numpy().astype(np.float32, copy=False)
        )

        graph_obj = torch_load_compat(rdkit_index.path_for(ligand_id))
        if not isinstance(graph_obj, dict) or "ue_x" not in graph_obj or "ue_adj" not in graph_obj:
            raise ValueError(f"Invalid RDKit UE graph artifact for ligand {ligand_id}")
        x = graph_obj["ue_x"]
        adj = graph_obj["ue_adj"]
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if not isinstance(adj, torch.Tensor):
            adj = torch.as_tensor(adj)
        x = x.detach().cpu().float()
        adj = adj.detach().cpu().float()
        if x.ndim != 2 or x.shape[0] <= 0 or x.shape[1] != int(expected_atom_dim):
            raise ValueError(
                f"Ligand {ligand_id} atom feature shape mismatch: {tuple(x.shape)}, "
                f"expected second dim={expected_atom_dim}"
            )
        if tuple(adj.shape) != (int(x.shape[0]), int(x.shape[0])):
            raise ValueError(f"Ligand {ligand_id} adjacency shape mismatch: {tuple(adj.shape)}")
        if not torch.isfinite(x).all() or not torch.isfinite(adj).all():
            raise ValueError(f"Ligand {ligand_id} graph contains NaN/Inf")
        if max_atoms is not None and int(max_atoms) > 0 and x.shape[0] > int(max_atoms):
            x = x[: int(max_atoms)]
            adj = adj[: int(max_atoms), : int(max_atoms)]
        ligand_graphs[ligand_id] = {
            "x": np.ascontiguousarray(x.numpy().astype(np.float32, copy=False)),
            "adj": np.ascontiguousarray(adj.numpy().astype(np.float32, copy=False)),
        }

    return ligand_embeddings, ligand_graphs


class ProteinStore:
    """Read-only ESMC residue embedding store backed by a manifest."""

    def __init__(
        self,
        index: ArtifactIndex,
        expected_dim: int = DEFAULT_PROTEIN_DIM,
        preload_hashes: Optional[Sequence[str]] = None,
        preload: bool = False,
        max_cache_items: int = 4096,
    ):
        self.index = index
        self.expected_dim = int(expected_dim)
        self.max_cache_items = int(max(1, max_cache_items))
        self.cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        if preload and preload_hashes is not None:
            ids = list(dict.fromkeys(normalize_entity_id(x) for x in preload_hashes))
            self.index.require(ids)
            for target_id in tqdm(ids, desc="[Protein] Preloading precomputed ESMC embeddings"):
                self.get(target_id)

    def _load_one(self, target_id: str) -> np.ndarray:
        obj = torch_load_compat(self.index.path_for(target_id))
        if not isinstance(obj, dict) or "residue_embedding" not in obj:
            raise ValueError(f"Invalid ESMC artifact for protein {target_id}")
        tensor = obj["residue_embedding"]
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
        tensor = tensor.detach().cpu().float()
        if tensor.ndim != 2 or tensor.shape[0] <= 0 or tensor.shape[1] != self.expected_dim:
            raise ValueError(
                f"Protein {target_id} ESMC shape mismatch: {tuple(tensor.shape)}, "
                f"expected [L,{self.expected_dim}]"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Protein {target_id} ESMC embedding contains NaN/Inf")
        return np.ascontiguousarray(tensor.numpy().astype(np.float32, copy=False))

    def get(self, target_id: str) -> np.ndarray:
        key = normalize_entity_id(target_id)
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        value = self._load_one(key)
        self.cache[key] = value
        while len(self.cache) > self.max_cache_items:
            self.cache.popitem(last=False)
        return value

    def get_length(self, target_id: str) -> int:
        return int(self.get(target_id).shape[0])


class ProteinGraphStore:
    """Read-only ESM2 contact graph store backed by embedding_generation manifests."""

    def __init__(
        self,
        index: ArtifactIndex,
        mode: str = "contact",
        seq_band_width: int = 2,
        contact_threshold: float = 0.0,
        max_cache_items: int = 4096,
    ):
        self.index = index
        self.mode = str(mode or "contact").lower()
        if self.mode == "auto":
            self.mode = "contact"
        if self.mode == "band":
            raise ValueError(
                "protein_graph_mode='band' is disabled. Use contact/contact_plus_band/hybrid."
            )
        if self.mode not in {"contact", "contact_plus_band", "hybrid"}:
            raise ValueError(
                f"Unsupported protein_graph_mode={mode}; allowed: contact/contact_plus_band/hybrid"
            )
        self.seq_band_width = int(max(1, seq_band_width))
        self.contact_threshold = float(max(0.0, contact_threshold))
        self.max_cache_items = int(max(1, max_cache_items))
        self.cache: "OrderedDict[Tuple[str, int], np.ndarray]" = OrderedDict()

    def _load_contact_adj(self, target_id: str, num_nodes: int) -> np.ndarray:
        path = self.index.path_for(target_id)
        try:
            with np.load(path, allow_pickle=False) as data:
                required = {"edge_index", "edge_weight", "num_nodes"}
                missing = required - set(data.files)
                if missing:
                    raise ValueError(f"Missing fields {sorted(missing)}")
                edge_index = np.asarray(data["edge_index"], dtype=np.int64)
                edge_weight = np.asarray(data["edge_weight"], dtype=np.float32).reshape(-1)
                graph_nodes = int(np.asarray(data["num_nodes"]).reshape(-1)[0])
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read precomputed ESM2 graph for protein {target_id}: {path}; {exc}"
            ) from exc

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"Protein {target_id} edge_index shape mismatch: {edge_index.shape}")
        if edge_weight.shape[0] != edge_index.shape[1]:
            raise ValueError(f"Protein {target_id} edge count/weight mismatch")
        if graph_nodes <= 0:
            raise ValueError(f"Protein {target_id} graph has invalid num_nodes={graph_nodes}")

        # Convert sparse contact edges to a normalized dense adjacency matrix.
        adj = dense_from_edge_index(
            edge_index=edge_index,
            edge_weight=edge_weight,
            num_nodes=graph_nodes,
        )
        if self.contact_threshold > 0:
            adj = adj.copy()
            mask = adj >= self.contact_threshold
            adj = adj * mask.astype(np.float32)
            np.fill_diagonal(adj, 1.0)
        # Resize safely instead of regenerating feature artifacts.
        return ensure_adj_size(adj, int(num_nodes))

    def get(self, target_id: str, num_nodes: int) -> np.ndarray:
        key = (normalize_entity_id(target_id), int(max(1, num_nodes)))
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value

        contact_adj = self._load_contact_adj(key[0], key[1])
        if self.mode in {"contact_plus_band", "hybrid"}:
            band_adj = build_band_protein_adjacency(
                key[1], band_width=self.seq_band_width
            )
            adj = normalize_dense_adjacency(np.maximum(contact_adj, band_adj))
        else:
            adj = contact_adj

        self.cache[key] = adj
        while len(self.cache) > self.max_cache_items:
            self.cache.popitem(last=False)
        return adj


def build_or_load_length_cache(
    target_ids: Sequence[str],
    protein_store: ProteinStore,
    cache_path: str,
) -> Dict[str, int]:
    cache = load_json(cache_path, default={}) or {}
    cache = {normalize_entity_id(k): int(v) for k, v in cache.items()}
    needed = list(dict.fromkeys(normalize_entity_id(x) for x in target_ids))
    missing = [x for x in needed if x not in cache]
    if missing:
        print(f"[Protein] Reading lengths for {len(missing)} proteins from precomputed ESMC artifacts...")
        protein_store.index.require(missing)
        for target_id in tqdm(missing, desc="[Protein] Reading ESMC lengths"):
            cache[target_id] = int(protein_store.get_length(target_id))
        save_json(cache, cache_path)
    return {x: int(cache[x]) for x in needed}


def gc_collect_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def prepare_scene_split_dataframe(split_paths: Dict[str, str]) -> pd.DataFrame:
    """Load standardized DTA split files with drug_id, target_id, affinity, and Split columns."""
    train_df = pd.read_csv(split_paths["train"])
    val_df = pd.read_csv(split_paths["val"])
    test_df = pd.read_csv(split_paths["test"])

    required_cols = {"drug_id", "target_id", "affinity", "Split"}
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{split_paths[name]} is missing columns: {sorted(missing)}")

    train_df = train_df[["drug_id", "target_id", "affinity", "Split"]].copy()
    val_df = val_df[["drug_id", "target_id", "affinity", "Split"]].copy()
    test_df = test_df[["drug_id", "target_id", "affinity", "Split"]].copy()

    for xdf in [train_df, val_df, test_df]:
        xdf["drug_id"] = xdf["drug_id"].apply(normalize_entity_id)
        xdf["target_id"] = xdf["target_id"].apply(normalize_entity_id)
        xdf["affinity"] = pd.to_numeric(xdf["affinity"], errors="coerce")
        xdf["Split"] = xdf["Split"].astype(str)
        xdf.dropna(subset=["affinity"], inplace=True)

    if set(train_df["Split"].unique()) - {"train"}:
        raise ValueError(f"{split_paths['train']} must contain only Split=train")
    if set(val_df["Split"].unique()) - {"val"}:
        raise ValueError(f"{split_paths['val']} must contain only Split=val")
    if set(test_df["Split"].unique()) - {"test"}:
        raise ValueError(f"{split_paths['test']} must contain only Split=test")

    df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)

    pair_split_n = (
        df.groupby(["drug_id", "target_id"])["Split"]
        .nunique()
        .reset_index(name="n_split")
    )
    conflicted = pair_split_n[pair_split_n["n_split"] > 1]
    if not conflicted.empty:
        raise ValueError(
            f"Found {len(conflicted)} drug-target pairs appearing in multiple splits "
            f"in {split_paths['scene_dir']}."
        )

    df = (
        df.groupby(["drug_id", "target_id", "Split"], as_index=False)["affinity"]
        .mean()
        .reset_index(drop=True)
    )
    return df

class DTADataset(Dataset):
    """Shared dataset wrapper for standardized DTA split files."""
    def __init__(
        self,
        df: pd.DataFrame,
        ligand_embeddings: Dict[str, np.ndarray],
        ligand_graphs: Dict[str, Dict[str, np.ndarray]],
        protein_store: ProteinStore,
        protein_graph_store: ProteinGraphStore,
        target_mean: float,
        target_std: float,
        max_protein_len: Optional[int] = None,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.ligand_embeddings = ligand_embeddings
        self.ligand_graphs = ligand_graphs
        self.protein_store = protein_store
        self.protein_graph_store = protein_graph_store
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)
        self.max_protein_len = max_protein_len

        self.ligand_ids = [normalize_entity_id(x) for x in self.df["drug_id"].tolist()]
        self.seq_hashes = [normalize_entity_id(x) for x in self.df["target_id"].tolist()]
        self.targets_raw = self.df["affinity"].astype(np.float32).to_numpy()
        self.targets_norm = ((self.targets_raw - self.target_mean) / self.target_std).astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        ligand_id = self.ligand_ids[idx]
        seq_hash = self.seq_hashes[idx]
        lig = self.ligand_embeddings[ligand_id]
        graph = self.ligand_graphs[ligand_id]

        prot_full = self.protein_store.get(seq_hash)
        full_len = int(prot_full.shape[0])
        keep_idx = make_downsample_indices(full_len, self.max_protein_len)
        prot = prot_full[keep_idx]

        protein_adj_full = self.protein_graph_store.get(seq_hash, num_nodes=full_len)
        protein_adj = protein_adj_full[np.ix_(keep_idx, keep_idx)].astype(np.float32, copy=False)


        return {
            "ligand": torch.from_numpy(lig).float(),
            "graph_x": torch.from_numpy(graph["x"]).float(),
            "graph_adj": torch.from_numpy(graph["adj"]).float(),
            "protein": torch.from_numpy(prot).float(),
            "protein_adj": torch.from_numpy(protein_adj).float(),
            "target_norm": torch.tensor(self.targets_norm[idx], dtype=torch.float32),
            "target_raw": torch.tensor(self.targets_raw[idx], dtype=torch.float32),
            "ligand_id": ligand_id,
            "seq_hash": seq_hash,
        }

class InMemoryProteinStore:
    """Read-only in-memory protein embedding store."""
    def __init__(self, protein_arrays: Dict[str, np.ndarray]):
        self.cache = {str(k): np.ascontiguousarray(v.astype(np.float32, copy=False)) for k, v in protein_arrays.items()}

    def get(self, seq_hash: str) -> np.ndarray:
        seq_hash = str(seq_hash)
        if seq_hash not in self.cache:
            raise KeyError(f"Protein embedding not preloaded in memory: {seq_hash}")
        return self.cache[seq_hash]

    def get_length(self, seq_hash: str) -> int:
        return int(self.get(seq_hash).shape[0])

class InMemoryProteinGraphStore:
    """Read-only in-memory protein graph store."""
    def __init__(self, protein_adj_arrays: Dict[str, np.ndarray]):
        self.cache = {str(k): np.ascontiguousarray(v.astype(np.float32, copy=False)) for k, v in protein_adj_arrays.items()}

    def get(self, seq_hash: str, num_nodes: int) -> np.ndarray:
        seq_hash = str(seq_hash)
        if seq_hash not in self.cache:
            raise KeyError(f"Protein graph not preloaded in memory: {seq_hash}")
        adj = self.cache[seq_hash]
        # Handle unexpected size mismatches safely.
        if adj.shape[0] != int(num_nodes) or adj.shape[1] != int(num_nodes):
            return ensure_adj_size(adj, int(num_nodes))
        return adj

def _format_bytes(num_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"

class PadCollate:
    def __call__(self, batch: List[dict]) -> dict:
        bs = len(batch)
        ligands = torch.stack([x["ligand"] for x in batch], dim=0)

        atom_lengths = [int(x["graph_x"].shape[0]) for x in batch]
        max_atoms = max(atom_lengths)
        atom_feat_dim = int(batch[0]["graph_x"].shape[1])
        graph_x = torch.zeros(bs, max_atoms, atom_feat_dim, dtype=torch.float32)
        graph_adj = torch.zeros(bs, max_atoms, max_atoms, dtype=torch.float32)
        graph_mask = torch.zeros(bs, max_atoms, dtype=torch.bool)
        for i, x in enumerate(batch):
            n = atom_lengths[i]
            graph_x[i, :n] = x["graph_x"]
            graph_adj[i, :n, :n] = x["graph_adj"]
            graph_mask[i, :n] = True

        prot_lengths = [int(x["protein"].shape[0]) for x in batch]
        max_len = max(prot_lengths)
        feat_dim = int(batch[0]["protein"].shape[1])
        proteins = torch.zeros(bs, max_len, feat_dim, dtype=torch.float32)
        protein_adj = torch.zeros(bs, max_len, max_len, dtype=torch.float32)
        mask = torch.zeros(bs, max_len, dtype=torch.bool)
        for i, x in enumerate(batch):
            l = prot_lengths[i]
            proteins[i, :l] = x["protein"]
            protein_adj[i, :l, :l] = x["protein_adj"]
            mask[i, :l] = True
        target_norm = torch.stack([x["target_norm"] for x in batch], dim=0)
        target_raw = torch.stack([x["target_raw"] for x in batch], dim=0)
        ligand_ids = [x["ligand_id"] for x in batch]
        seq_hashes = [x["seq_hash"] for x in batch]
        return {
            "ligand": ligands,
            "graph_x": graph_x,
            "graph_adj": graph_adj,
            "graph_mask": graph_mask,
            "protein": proteins,
            "protein_adj": protein_adj,
            "mask": mask,
            "target_norm": target_norm,
            "target_raw": target_raw,
            "lengths": torch.tensor(prot_lengths, dtype=torch.long),
            "atom_lengths": torch.tensor(atom_lengths, dtype=torch.long),
            "ligand_ids": ligand_ids,
            "seq_hashes": seq_hashes,
        }

class SortishBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        bucket_size_multiplier: int = 50,
    ):
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.bucket_size_multiplier = int(max(2, bucket_size_multiplier))
        self.indices = np.arange(len(self.lengths))

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return math.ceil(len(self.lengths) / self.batch_size)

    def __iter__(self):
        if len(self.indices) == 0:
            return
            yield
        if self.shuffle:
            perm = np.random.permutation(self.indices)
        else:
            perm = self.indices.copy()

        megabatch_size = self.batch_size * self.bucket_size_multiplier
        batches = []
        for chunk in chunked(perm, megabatch_size):
            chunk = np.asarray(chunk)
            order = np.argsort(self.lengths[chunk])[::-1]
            chunk_sorted = chunk[order]
            for batch in chunked(chunk_sorted, self.batch_size):
                batch = list(map(int, batch))
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        if self.shuffle and len(batches) > 1:
            batch_max_lens = [max(self.lengths[b]) for b in batches]
            first_idx = int(np.argmax(batch_max_lens))
            first_batch = batches[first_idx]
            rest = batches[:first_idx] + batches[first_idx + 1:]
            random.shuffle(rest)
            batches = [first_batch] + rest

        for batch in batches:
            yield batch

class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)

def sanitize_tensor(x: torch.Tensor, clamp_value: Optional[float] = None) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
    if clamp_value is not None and clamp_value > 0:
        x = x.clamp(min=-clamp_value, max=clamp_value)
    return x

class GatedAttentionPooling(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_safe = sanitize_tensor(x)
        mask = mask.bool()
        row_has_token = mask.any(dim=-1, keepdim=True)

        scores = self.score(x_safe.float()).squeeze(-1)
        scores = sanitize_tensor(scores, clamp_value=50.0)
        scores = scores.masked_fill(~mask, -1e4)

        attn = torch.softmax(scores, dim=-1)
        attn = torch.where(row_has_token, attn, torch.zeros_like(attn))
        attn = attn * mask.to(attn.dtype)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        attn = sanitize_tensor(attn, clamp_value=1.0)

        pooled = torch.bmm(attn.unsqueeze(1), x_safe.float()).squeeze(1)
        pooled = torch.where(row_has_token, pooled, torch.zeros_like(pooled))
        pooled = sanitize_tensor(pooled, clamp_value=1e4)
        return pooled.to(x.dtype), attn.to(x.dtype)

class GraphMessagePassingBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.msg_proj = nn.Linear(dim, dim)
        self.self_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_safe = sanitize_tensor(x)
        adj_safe = sanitize_tensor(adj, clamp_value=5.0)
        h = self.norm1(x_safe)
        neigh = torch.bmm(adj_safe, self.msg_proj(h))
        self_term = self.self_proj(h)
        x_safe = x_safe + self.dropout(F.gelu(neigh + self_term))
        x_safe = x_safe + self.ffn(self.norm2(x_safe))
        x_safe = sanitize_tensor(x_safe, clamp_value=1e4)
        x_safe = x_safe * mask.unsqueeze(-1).to(x_safe.dtype)
        return x_safe

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query_tokens: torch.Tensor,
        query_mask: torch.Tensor,
        context_tokens: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        query_tokens = sanitize_tensor(query_tokens)
        context_tokens = sanitize_tensor(context_tokens)
        query_mask = query_mask.bool()
        context_mask = context_mask.bool()

        attn_out = torch.zeros_like(query_tokens)
        valid_rows = query_mask.any(dim=1) & context_mask.any(dim=1)
        if bool(valid_rows.any()):
            q = self.q_norm(query_tokens[valid_rows]).float()
            kv = self.kv_norm(context_tokens[valid_rows]).float()
            kpm = ~context_mask[valid_rows]
            with torch.amp.autocast("cuda", enabled=False) if q.device.type == "cuda" else nullcontext():
                attn_valid, _ = self.attn(
                    q,
                    kv,
                    kv,
                    key_padding_mask=kpm,
                    need_weights=False,
                )
            attn_valid = sanitize_tensor(attn_valid, clamp_value=1e4)
            attn_out[valid_rows] = attn_valid.to(query_tokens.dtype)

        x = query_tokens + self.dropout(attn_out)
        x = x + self.ffn(self.ffn_norm(x))
        x = sanitize_tensor(x, clamp_value=1e4)
        x = x * query_mask.unsqueeze(-1).to(x.dtype)
        return x

class ResidueSelector(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def forward(self, protein_tokens: torch.Tensor, protein_mask: torch.Tensor, drug_global: torch.Tensor) -> torch.Tensor:
        d = drug_global.unsqueeze(1).expand(-1, protein_tokens.size(1), -1)
        feat = torch.cat([protein_tokens, d, protein_tokens * d], dim=-1)
        scores = self.proj(feat).squeeze(-1)
        scores = scores.masked_fill(~protein_mask.bool(), -1e4)
        return sanitize_tensor(scores, 50.0)

def batched_topk_gather(tokens: torch.Tensor, scores: torch.Tensor, mask: torch.Tensor, topk: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Gather top-k valid tokens from [B, L, D].
    Returns selected_tokens [B, K, D], selected_mask [B, K], selected_idx [B, K].
    """
    B, L, D = tokens.shape
    valid_counts = mask.long().sum(dim=1)
    k_eff = min(int(topk), int(valid_counts.max().item())) if B > 0 else int(topk)
    k_eff = max(1, k_eff)
    masked_scores = scores.masked_fill(~mask.bool(), -1e4)
    top_vals, top_idx = torch.topk(masked_scores, k=k_eff, dim=1)
    selected = torch.gather(tokens, 1, top_idx.unsqueeze(-1).expand(-1, -1, D))
    selected_mask = top_vals > -1e3
    return selected, selected_mask, top_idx

class SparseInteractionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.drug_from_selected_protein = CrossAttentionBlock(dim, num_heads, dropout)
        self.selected_protein_from_drug = CrossAttentionBlock(dim, num_heads, dropout)

    def forward(self, drug_tokens: torch.Tensor, drug_mask: torch.Tensor, selected_protein_tokens: torch.Tensor, selected_protein_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        next_drug = self.drug_from_selected_protein(drug_tokens, drug_mask, selected_protein_tokens, selected_protein_mask)
        next_selected_protein = self.selected_protein_from_drug(selected_protein_tokens, selected_protein_mask, drug_tokens, drug_mask)
        return next_drug, next_selected_protein

class UEAlignNet12A(nn.Module):
    def __init__(
        self,
        ligand_dim: int = DEFAULT_LIGAND_DIM,
        graph_in_dim: int = DEFAULT_ATOM_FEAT_DIM,
        protein_dim: int = DEFAULT_PROTEIN_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        dropout: float = 0.1,
        num_graph_layers: int = 3,
        num_protein_graph_layers: int = 2,
        num_sparse_interaction_layers: int = 2,
        num_attention_heads: int = 8,
        residue_topk: int = 128,
        use_ligand_global: bool = True,
    ):
        super().__init__()
        if hidden_dim % num_attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_attention_heads")
        self.residue_topk = int(max(1, residue_topk))
        self.use_ligand_global = bool(use_ligand_global)

        self.ligand_norm = nn.LayerNorm(ligand_dim)
        self.ligand_encoder = nn.Sequential(
            MLPBlock(ligand_dim, 1024, hidden_dim, dropout),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.graph_in_norm = nn.LayerNorm(graph_in_dim)
        self.graph_encoder_in = nn.Linear(graph_in_dim, hidden_dim)
        self.graph_layers = nn.ModuleList([GraphMessagePassingBlock(hidden_dim, dropout) for _ in range(max(1, num_graph_layers))])

        self.protein_seq_norm = nn.LayerNorm(protein_dim)
        self.protein_seq_encoder = nn.Sequential(
            nn.Linear(protein_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.protein_graph_norm = nn.LayerNorm(protein_dim)
        self.protein_graph_encoder_in = nn.Linear(protein_dim, hidden_dim)
        self.protein_graph_layers = nn.ModuleList([GraphMessagePassingBlock(hidden_dim, dropout) for _ in range(max(1, num_protein_graph_layers))])

        self.graph_pool = GatedAttentionPooling(hidden_dim, dropout)
        self.protein_pool = GatedAttentionPooling(hidden_dim, dropout)
        self.drug_pool = GatedAttentionPooling(hidden_dim, dropout)
        self.selected_protein_pool = GatedAttentionPooling(hidden_dim, dropout)

        self.global_fuse_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.protein_branch_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.selector = ResidueSelector(hidden_dim, dropout=dropout)
        self.sparse_layers = nn.ModuleList([SparseInteractionBlock(hidden_dim, num_attention_heads, dropout) for _ in range(max(1, num_sparse_interaction_layers))])

        fusion_dim = hidden_dim * 8
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, ligand_global: torch.Tensor, graph_x: torch.Tensor, graph_adj: torch.Tensor, graph_mask: torch.Tensor, protein: torch.Tensor, protein_adj: torch.Tensor, protein_mask: torch.Tensor) -> torch.Tensor:
        ligand_global = sanitize_tensor(ligand_global, 1e4)
        graph_x = sanitize_tensor(graph_x, 1e4)
        graph_adj = sanitize_tensor(graph_adj, 5.0)
        protein = sanitize_tensor(protein, 1e4)
        protein_adj = sanitize_tensor(protein_adj, 5.0)
        graph_mask = graph_mask.bool()
        protein_mask = protein_mask.bool()

        lig_global = self.ligand_encoder(self.ligand_norm(ligand_global))

        drug_tokens = self.graph_encoder_in(self.graph_in_norm(graph_x))
        drug_tokens = drug_tokens * graph_mask.unsqueeze(-1).to(drug_tokens.dtype)
        for layer in self.graph_layers:
            drug_tokens = layer(drug_tokens, graph_adj, graph_mask)
        graph_global, _ = self.graph_pool(drug_tokens, graph_mask)

        if self.use_ligand_global:
            fuse_gate = self.global_fuse_gate(torch.cat([lig_global, graph_global], dim=-1))
            fused_drug_global = fuse_gate * lig_global + (1.0 - fuse_gate) * graph_global
        else:
            fused_drug_global = graph_global
        drug_tokens = drug_tokens + fused_drug_global.unsqueeze(1)
        drug_tokens = drug_tokens * graph_mask.unsqueeze(-1).to(drug_tokens.dtype)

        protein_seq_tokens = self.protein_seq_encoder(self.protein_seq_norm(protein))
        protein_seq_tokens = protein_seq_tokens * protein_mask.unsqueeze(-1).to(protein_seq_tokens.dtype)

        protein_graph_tokens = self.protein_graph_encoder_in(self.protein_graph_norm(protein))
        protein_graph_tokens = protein_graph_tokens * protein_mask.unsqueeze(-1).to(protein_graph_tokens.dtype)
        for layer in self.protein_graph_layers:
            protein_graph_tokens = layer(protein_graph_tokens, protein_adj, protein_mask)

        protein_gate = self.protein_branch_gate(torch.cat([protein_seq_tokens, protein_graph_tokens], dim=-1))
        protein_tokens = protein_gate * protein_seq_tokens + (1.0 - protein_gate) * protein_graph_tokens
        protein_tokens = sanitize_tensor(protein_tokens, 1e4)
        protein_tokens = protein_tokens * protein_mask.unsqueeze(-1).to(protein_tokens.dtype)

        protein_global, _ = self.protein_pool(protein_tokens, protein_mask)

        residue_scores = self.selector(protein_tokens, protein_mask, fused_drug_global)
        selected_protein_tokens, selected_protein_mask, _ = batched_topk_gather(protein_tokens, residue_scores, protein_mask, self.residue_topk)

        for layer in self.sparse_layers:
            drug_tokens, selected_protein_tokens = layer(drug_tokens, graph_mask, selected_protein_tokens, selected_protein_mask)

        drug_local, _ = self.drug_pool(drug_tokens, graph_mask)
        protein_selected_local, _ = self.selected_protein_pool(selected_protein_tokens, selected_protein_mask)

        fusion = torch.cat(
            [
                fused_drug_global,
                drug_local,
                protein_global,
                protein_selected_local,
                fused_drug_global * protein_selected_local,
                torch.abs(fused_drug_global - protein_selected_local),
                drug_local * protein_selected_local,
                torch.abs(drug_local - protein_selected_local),
            ],
            dim=-1,
        )
        out = self.head(sanitize_tensor(fusion, 1e4)).squeeze(-1)
        return sanitize_tensor(out, 1e4)

def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])

def spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    true_rank = pd.Series(y_true).rank(method="average").to_numpy()
    pred_rank = pd.Series(y_pred).rank(method="average").to_numpy()
    return pearson_corr(true_rank, pred_rank)

def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)

class _FenwickTree:
    def __init__(self, size: int):
        self.size = int(max(1, size))
        self.tree = np.zeros(self.size + 1, dtype=np.float64)

    def add(self, idx: int, value: float) -> None:
        idx = int(idx) + 1
        while idx <= self.size:
            self.tree[idx] += value
            idx += idx & -idx

    def prefix_sum(self, idx: int) -> float:
        if idx < 0:
            return 0.0
        idx = int(idx) + 1
        out = 0.0
        while idx > 0:
            out += float(self.tree[idx])
            idx -= idx & -idx
        return out

def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the concordance index over pairs with unequal observed affinities."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = int(len(y_true))
    if n < 2:
        return float("nan")

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    n = int(len(y_true))
    if n < 2:
        return float("nan")

    order = np.argsort(y_true, kind="mergesort")
    y_true = y_true[order]
    y_pred = y_pred[order]

    pred_values = np.unique(y_pred)
    pred_ranks = np.searchsorted(pred_values, y_pred, side="left")
    ft = _FenwickTree(len(pred_values))

    concordant = 0.0
    comparable = 0.0
    seen_count = 0.0
    i = 0
    while i < n:
        j = i + 1
        while j < n and y_true[j] == y_true[i]:
            j += 1

        # Compare each tie group only against samples with smaller observed values.
        for k in range(i, j):
            r = int(pred_ranks[k])
            less = ft.prefix_sum(r - 1)
            equal = ft.prefix_sum(r) - ft.prefix_sum(r - 1)
            concordant += less + 0.5 * equal
            comparable += seen_count

        for k in range(i, j):
            ft.add(int(pred_ranks[k]), 1.0)
            seen_count += 1.0
        i = j

    if comparable <= 0:
        return float("nan")
    return float(concordant / comparable)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mse = float(np.mean((y_pred - y_true) ** 2))
    rmse = float(np.sqrt(mse))
    return {
        "n": int(len(y_true)),
        "mse": mse,
        "ci": concordance_index(y_true, y_pred),
        "rm2": rm2_score(y_true, y_pred),
        "rmse": rmse,
        "mae": float(np.mean(np.abs(y_pred - y_true))),
        "r2": r2_score_np(y_true, y_pred),
        "pearson": pearson_corr(y_true, y_pred),
        "spearman": spearman_corr(y_true, y_pred),
    }

def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out

def build_scheduler(optimizer, total_steps: int, warmup_steps: int):
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

class RegressionRankingLoss(nn.Module):
    def __init__(self, huber_beta: float = 1.0, rank_weight: float = 0.2):
        super().__init__()
        self.reg_loss = nn.SmoothL1Loss(beta=huber_beta)
        self.rank_weight = float(rank_weight)

    def pairwise_rank_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        group_ids: Sequence[str],
    ) -> torch.Tensor:
        total_loss = pred.sum() * 0.0
        valid_groups = 0
        unique_groups = list(dict.fromkeys(group_ids))
        for gid in unique_groups:
            idx = [i for i, g in enumerate(group_ids) if g == gid]
            if len(idx) < 2:
                continue
            p = pred[idx]
            y = target[idx]
            diff_y = y[:, None] - y[None, :]
            valid = torch.triu(diff_y.abs() > 1e-6, diagonal=1)
            if not bool(valid.any()):
                continue
            sign = torch.sign(diff_y)
            diff_p = p[:, None] - p[None, :]
            pair_loss = F.softplus(-sign * diff_p)
            total_loss = total_loss + pair_loss[valid].mean()
            valid_groups += 1
        if valid_groups == 0:
            return pred.sum() * 0.0
        return total_loss / float(valid_groups)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        group_ids: Sequence[str],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        reg = self.reg_loss(pred, target)
        rank = self.pairwise_rank_loss(pred, target, group_ids)
        total = reg + self.rank_weight * rank
        parts = {"reg_loss": reg.detach(), "rank_loss": rank.detach(), "total_loss": total.detach()}
        return total, parts

def compute_checkpoint_score(metrics: dict, spearman_weight: float = 0.05) -> float:
    spearman = float(metrics.get("spearman", float("nan")))
    if not np.isfinite(spearman):
        spearman = -1.0
    return float(-metrics["rmse"] + spearman_weight * spearman)

def should_save_best(
    metrics: dict,
    best_val_rmse: float,
    best_val_spearman: float,
    rmse_min_delta: float,
    rmse_tolerance: float,
    spearman_min_delta: float,
) -> bool:
    cur_rmse = float(metrics["rmse"])
    cur_spearman = float(metrics.get("spearman", float("nan")))
    if not np.isfinite(cur_spearman):
        cur_spearman = -1.0
    best_spearman_safe = best_val_spearman if np.isfinite(best_val_spearman) else -1.0

    if cur_rmse < best_val_rmse - rmse_min_delta:
        return True
    if cur_rmse <= best_val_rmse + rmse_tolerance and cur_spearman > best_spearman_safe + spearman_min_delta:
        return True
    return False

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    scheduler,
    scaler,
    loss_fn,
    device: torch.device,
    grad_clip: float,
    use_amp: bool = True,
    desc: Optional[str] = None,
) -> dict:
    model.train()
    running_total = 0.0
    running_reg = 0.0
    running_rank = 0.0
    n_samples = 0
    skipped_batches = 0

    autocast_enabled = bool(use_amp and device.type == "cuda")
    iterator = tqdm(loader, desc=desc, leave=False) if desc else loader

    for batch in iterator:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=autocast_enabled) if device.type == "cuda" else nullcontext():
            pred_norm = model(
                batch["ligand"],
                batch["graph_x"],
                batch["graph_adj"],
                batch["graph_mask"],
                batch["protein"],
                batch["protein_adj"],
                batch["mask"],
            )
            pred_norm = sanitize_tensor(pred_norm, clamp_value=1e4)
            target_norm = sanitize_tensor(batch["target_norm"], clamp_value=1e4)
            loss, parts = loss_fn(pred_norm, target_norm, batch["seq_hashes"])

        if not torch.isfinite(loss):
            skipped_batches += 1
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.update()
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip if grad_clip is not None and grad_clip > 0 else 1e9,
                error_if_nonfinite=False,
            )
            if not torch.isfinite(total_grad_norm):
                skipped_batches += 1
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip if grad_clip is not None and grad_clip > 0 else 1e9,
                error_if_nonfinite=False,
            )
            if not torch.isfinite(total_grad_norm):
                skipped_batches += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        batch_size = int(batch["ligand"].size(0))
        running_total += float(parts["total_loss"].item()) * batch_size
        running_reg += float(parts["reg_loss"].item()) * batch_size
        running_rank += float(parts["rank_loss"].item()) * batch_size
        n_samples += batch_size

    denom = max(1, n_samples)
    return {
        "loss": running_total / denom if n_samples > 0 else float("nan"),
        "reg_loss": running_reg / denom if n_samples > 0 else float("nan"),
        "rank_loss": running_rank / denom if n_samples > 0 else float("nan"),
        "skipped_batches": int(skipped_batches),
    }

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_val_rmse: float,
    best_val_spearman: float,
    best_val_score: float,
    patience_counter: int,
    train_mean: float,
    train_std: float,
    split_md5: str,
    seed: int,
    scene: str,
    config: dict,
) -> None:
    payload = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_val_rmse": float(best_val_rmse),
        "best_val_spearman": float(best_val_spearman),
        "best_val_score": float(best_val_score),
        "patience_counter": int(patience_counter),
        "train_mean": float(train_mean),
        "train_std": float(train_std),
        "split_md5": split_md5,
        "seed": int(seed),
        "scene": scene,
        "config": config,
    }
    ensure_dir(os.path.dirname(path))
    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)

def load_checkpoint(path: str, device: torch.device) -> dict:
    return torch.load(path, map_location=device)

def get_scene_split_paths(splits_dir: str, seed: int, scene: str) -> Dict[str, str]:
    scene_dir = os.path.join(splits_dir, f"seed_{seed:04d}", scene)
    return {
        "scene_dir": scene_dir,
        "train": os.path.join(scene_dir, "train.csv"),
        "val": os.path.join(scene_dir, "val.csv"),
        "test": os.path.join(scene_dir, "test.csv"),
    }

def build_dataloader(
    df: pd.DataFrame,
    ligand_embeddings: Dict[int, np.ndarray],
    ligand_graphs: Dict[int, Dict[str, np.ndarray]],
    protein_store: ProteinStore,
    protein_graph_store: ProteinGraphStore,
    train_mean: float,
    train_std: float,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    max_protein_len: Optional[int],
    lengths_lookup: Dict[str, int],
    shuffle: bool,
) -> DataLoader:
    dataset = DTADataset(
        df=df,
        ligand_embeddings=ligand_embeddings,
        ligand_graphs=ligand_graphs,
        protein_store=protein_store,
        protein_graph_store=protein_graph_store,
        target_mean=train_mean,
        target_std=train_std,
        max_protein_len=max_protein_len,
    )
    lengths = []
    for seq_hash in dataset.seq_hashes:
        length = int(lengths_lookup.get(seq_hash, 1))
        if max_protein_len is not None and max_protein_len > 0:
            length = min(length, max_protein_len)
        lengths.append(length)

    collate_fn = PadCollate()

    if shuffle:
        batch_sampler = SortishBatchSampler(lengths, batch_size=batch_size, shuffle=True, drop_last=False)
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            persistent_workers=bool(num_workers > 0),
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            persistent_workers=bool(num_workers > 0),
        )
    return loader

def summarize_split_counts(df: pd.DataFrame) -> dict:
    return {
        "train": int((df["Split"] == "train").sum()),
        "val": int((df["Split"] == "val").sum()),
        "test": int((df["Split"] == "test").sum()),
    }

def evaluate_and_save(
    model: nn.Module,
    split_name: str,
    loader: DataLoader,
    device: torch.device,
    train_mean: float,
    train_std: float,
    out_dir: str,
    use_amp: bool = True,
) -> dict:
    pred_df, metrics = predict_epoch(model, loader, device, train_mean, train_std, use_amp=use_amp)
    pred_df.to_csv(os.path.join(out_dir, f"predictions_{split_name}.csv"), index=False)
    save_json(metrics, os.path.join(out_dir, f"metrics_{split_name}.json"))
    print(
        f"[Eval][{split_name}] n={metrics['n']} "
        f"RMSE={metrics['rmse']:.4f} MSE={metrics['mse']:.4f} "
        f"R2={metrics['r2']:.4f} CI={metrics['ci']:.4f} "
        f"Pearson={metrics['pearson']:.4f} Spearman={metrics['spearman']:.4f}"
    )
    return metrics

def format_metrics(metrics: dict) -> str:
    return (
        f"RMSE={metrics['rmse']:.4f}, "
        f"MSE={metrics['mse']:.4f}, "
        f"R2={metrics['r2']:.4f}, "
        f"CI={metrics['ci']:.4f}, "
        f"Pearson={metrics['pearson']:.4f}, "
        f"Spearman={metrics['spearman']:.4f}"
    )

def get_scene_runtime_config(scene: str, args) -> dict:
    cfg = {
        "lr": float(args.lr),
        "max_epochs": int(args.max_epochs),
        "patience": int(args.patience),
        "warmup_epochs": int(args.warmup_epochs),
        "ckpt_rmse_tolerance": float(args.ckpt_rmse_tolerance),
        "rank_loss_weight": float(args.rank_loss_weight),
    }
    if getattr(args, "scene_aware_training", True):
        if scene == "cold_target":
            cfg["lr"] = float(args.cold_target_lr)
            cfg["max_epochs"] = int(args.cold_target_max_epochs)
            cfg["patience"] = int(args.cold_target_patience)
            cfg["warmup_epochs"] = int(args.cold_target_warmup_epochs)
            cfg["ckpt_rmse_tolerance"] = float(args.cold_target_ckpt_rmse_tolerance)
            cfg["rank_loss_weight"] = float(args.cold_target_rank_loss_weight)
        elif scene == "double_cold":
            cfg["lr"] = float(args.double_cold_lr)
            cfg["max_epochs"] = int(args.double_cold_max_epochs)
            cfg["patience"] = int(args.double_cold_patience)
            cfg["warmup_epochs"] = int(args.double_cold_warmup_epochs)
            cfg["ckpt_rmse_tolerance"] = float(args.double_cold_ckpt_rmse_tolerance)
            cfg["rank_loss_weight"] = float(args.double_cold_rank_loss_weight)
    return cfg

def safe_load_checkpoint_if_match(path: str, device: torch.device, split_md5: str):
    if not os.path.exists(path):
        return None
    try:
        ckpt = load_checkpoint(path, device=device)
    except Exception as exc:
        print(f"[Resume] Failed to load checkpoint {path}: {exc}")
        return None
    if ckpt.get("split_md5") != split_md5:
        raise RuntimeError(
            f"Checkpoint split md5 mismatch for {path}. Expected {split_md5}, got {ckpt.get('split_md5')}"
        )
    return ckpt

def choose_resume_checkpoint(best_ckpt_path: str, last_ckpt_path: str, status: dict, device: torch.device, split_md5: str):
    if status.get("completed", False) and os.path.exists(best_ckpt_path):
        ckpt = safe_load_checkpoint_if_match(best_ckpt_path, device, split_md5)
        if ckpt is not None:
            return "completed_best", ckpt

    best_ckpt = safe_load_checkpoint_if_match(best_ckpt_path, device, split_md5) if os.path.exists(best_ckpt_path) else None
    last_ckpt = safe_load_checkpoint_if_match(last_ckpt_path, device, split_md5) if os.path.exists(last_ckpt_path) else None

    if best_ckpt is None and last_ckpt is None:
        return None, None
    if best_ckpt is None:
        return "resume_last", last_ckpt
    if last_ckpt is None:
        return "resume_best_only", best_ckpt

    best_epoch = int(best_ckpt.get("epoch", -1))
    last_epoch = int(last_ckpt.get("epoch", -1))
    if last_epoch >= best_epoch:
        return "resume_last", last_ckpt
    return "resume_best_only", best_ckpt

def update_global_progress(model_root: str, payload: dict) -> None:
    save_json(payload, os.path.join(model_root, "run_progress.json"))

@torch.no_grad()
def predict_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    train_mean: float,
    train_std: float,
    use_amp: bool = True,
    desc: Optional[str] = None,
) -> Tuple[pd.DataFrame, dict]:
    model.eval()
    y_true_all = []
    y_pred_all = []
    ligand_ids_all = []
    seq_hashes_all = []

    autocast_enabled = bool(use_amp and device.type == "cuda")
    iterator = tqdm(loader, desc=desc, leave=False) if desc else loader

    for batch in iterator:
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast("cuda", enabled=autocast_enabled) if device.type == "cuda" else nullcontext():
            pred_norm = model(
                batch["ligand"],
                batch["graph_x"],
                batch["graph_adj"],
                batch["graph_mask"],
                batch["protein"],
                batch["protein_adj"],
                batch["mask"],
            )
        pred = pred_norm.float().cpu().numpy() * train_std + train_mean
        y_true = batch["target_raw"].float().cpu().numpy()

        y_true_all.append(y_true)
        y_pred_all.append(pred)
        ligand_ids_all.extend(batch["ligand_ids"])
        seq_hashes_all.extend(batch["seq_hashes"])

    y_true_all = np.concatenate(y_true_all, axis=0) if y_true_all else np.array([], dtype=np.float32)
    y_pred_all = np.concatenate(y_pred_all, axis=0) if y_pred_all else np.array([], dtype=np.float32)
    metrics = compute_metrics(y_true_all, y_pred_all)
    pred_df = pd.DataFrame(
        {
            "drug_id": ligand_ids_all,
            "target_id": seq_hashes_all,
            "y_true_affinity": y_true_all,
            "y_pred_affinity": y_pred_all,
        }
    )
    return pred_df, metrics

def collect_master_split_dataframe(args) -> pd.DataFrame:
    """Collect all entities required by the selected seeds and scenarios."""
    dfs = []
    for seed in args.seeds:
        for scene in args.target_scenarios:
            split_paths = get_scene_split_paths(args.splits_dir, seed, scene)
            dfs.append(prepare_scene_split_dataframe(split_paths))
    if not dfs:
        return pd.DataFrame(columns=["drug_id", "target_id", "Split", "affinity"])
    df = pd.concat(dfs, ignore_index=True)
    df = df.groupby(["drug_id", "target_id"], as_index=False)["affinity"].mean().reset_index(drop=True)
    return df

def preload_protein_embeddings_and_graphs_to_memory(
    args,
    unique_targets: Sequence[str],
    lengths_lookup: Dict[str, int],
    protein_index: ArtifactIndex,
    protein_graph_index: ArtifactIndex,
) -> Tuple[InMemoryProteinStore, InMemoryProteinGraphStore, Dict[str, int]]:
    """Load already-generated protein artifacts once; never generate them here."""
    unique_targets = [normalize_entity_id(x) for x in dict.fromkeys(unique_targets)]
    max_len = getattr(args, "max_protein_len", None)
    if max_len in (0, None):
        max_len = None

    if max_len is None and not getattr(args, "allow_full_length_memory_preload", False):
        raise RuntimeError(
            "preload_all_data_to_memory is enabled while max_protein_len is 0/None.\n"
            "This may load full LxL protein graphs into RAM. Use --max_protein_len 512/1024/1500, "
            "or explicitly pass --allow_full_length_memory_preload."
        )

    print("📦 [MemoryPreload] Loading precomputed ESMC embeddings and ESM2 graphs into memory...")
    print(
        f"📦 [MemoryPreload] targets={len(unique_targets)} | "
        f"max_protein_len={max_len if max_len is not None else 'FULL'}"
    )

    protein_index.require(unique_targets)
    protein_graph_index.require(unique_targets)
    disk_protein_store = ProteinStore(
        index=protein_index,
        expected_dim=args.protein_dim,
        preload=False,
        max_cache_items=max(
            128, min(int(getattr(args, "protein_cache_size", 4096)), len(unique_targets) + 8)
        ),
    )
    disk_graph_store = ProteinGraphStore(
        index=protein_graph_index,
        mode=args.protein_graph_mode,
        seq_band_width=args.protein_seq_band_width,
        contact_threshold=args.protein_contact_threshold,
        max_cache_items=max(
            128,
            min(int(getattr(args, "protein_graph_cache_size", 4096)), len(unique_targets) + 8),
        ),
    )

    protein_arrays: Dict[str, np.ndarray] = {}
    protein_adj_arrays: Dict[str, np.ndarray] = {}
    new_lengths_lookup: Dict[str, int] = dict(lengths_lookup)
    protein_bytes = 0
    graph_bytes = 0
    longest_before = 0
    longest_after = 0

    for target_id in tqdm(unique_targets, desc="[MemoryPreload] ESMC + ESM2 graph"):
        prot_full = disk_protein_store.get(target_id)
        full_len = int(prot_full.shape[0])
        longest_before = max(longest_before, full_len)
        keep_idx = make_downsample_indices(full_len, max_len)
        protein_adj_full = disk_graph_store.get(target_id, num_nodes=full_len)

        prot = np.ascontiguousarray(prot_full[keep_idx].astype(np.float32, copy=False))
        adj = np.ascontiguousarray(
            protein_adj_full[np.ix_(keep_idx, keep_idx)].astype(np.float32, copy=False)
        )
        protein_arrays[target_id] = prot
        protein_adj_arrays[target_id] = adj
        new_lengths_lookup[target_id] = int(prot.shape[0])
        longest_after = max(longest_after, int(prot.shape[0]))
        protein_bytes += int(prot.nbytes)
        graph_bytes += int(adj.nbytes)

    del disk_protein_store, disk_graph_store
    gc_collect_cuda()
    print(
        "✅ [MemoryPreload] done | "
        f"protein≈{_format_bytes(protein_bytes)} | graph≈{_format_bytes(graph_bytes)} | "
        f"total≈{_format_bytes(protein_bytes + graph_bytes)} | "
        f"longest {longest_before}->{longest_after}"
    )
    return (
        InMemoryProteinStore(protein_arrays),
        InMemoryProteinGraphStore(protein_adj_arrays),
        new_lengths_lookup,
    )


def prepare_dataset_runtime(args) -> dict:
    """Load and validate all precomputed artifacts required by selected splits."""
    print("📦 Preparing runtime from standardized splits and feature manifests...")
    master_df = collect_master_split_dataframe(args)
    unique_drugs = [
        normalize_entity_id(x) for x in sorted(set(master_df["drug_id"].tolist()))
    ]
    unique_targets = [
        normalize_entity_id(x) for x in sorted(set(master_df["target_id"].tolist()))
    ]
    if not unique_drugs or not unique_targets:
        raise RuntimeError(f"No entities found in selected {args.dataset_name} splits")

    embedding_root = Path(args.embedding_root)
    unimol_index = ArtifactIndex(embedding_root, args.dataset_name, "ligand", "unimol2")
    rdkit_index = ArtifactIndex(embedding_root, args.dataset_name, "ligand", "rdkit")
    protein_index = ArtifactIndex(embedding_root, args.dataset_name, "protein", "esmc")
    protein_graph_index = ArtifactIndex(
        embedding_root, args.dataset_name, "protein", "esm2_contact_graph"
    )

    # Every selected entity must already have valid feature artifacts.
    unimol_index.require(unique_drugs)
    rdkit_index.require(unique_drugs)
    protein_index.require(unique_targets)
    protein_graph_index.require(unique_targets)

    ligand_embeddings, ligand_graphs = load_ligand_features_from_manifests(
        ligand_ids=unique_drugs,
        unimol_index=unimol_index,
        rdkit_index=rdkit_index,
        expected_dim=args.ligand_dim,
        expected_atom_dim=DEFAULT_ATOM_FEAT_DIM,
        max_atoms=args.max_ligand_atoms,
    )

    disk_protein_store = ProteinStore(
        index=protein_index,
        expected_dim=args.protein_dim,
        preload_hashes=unique_targets if args.preload_proteins else None,
        preload=args.preload_proteins,
        max_cache_items=args.protein_cache_size,
    )
    length_cache_path = os.path.join(args.model_root, "protein_lengths_cache.json")
    lengths_lookup = build_or_load_length_cache(
        unique_targets, disk_protein_store, length_cache_path
    )

    if getattr(args, "preload_all_data_to_memory", False):
        protein_store, protein_graph_store, lengths_lookup = (
            preload_protein_embeddings_and_graphs_to_memory(
                args=args,
                unique_targets=unique_targets,
                lengths_lookup=lengths_lookup,
                protein_index=protein_index,
                protein_graph_index=protein_graph_index,
            )
        )
    else:
        protein_store = disk_protein_store
        protein_graph_store = ProteinGraphStore(
            index=protein_graph_index,
            mode=args.protein_graph_mode,
            seq_band_width=args.protein_seq_band_width,
            contact_threshold=args.protein_contact_threshold,
            max_cache_items=args.protein_graph_cache_size,
        )

    print(
        f"✅ [{args.dataset_name}] feature validation passed | "
        f"drugs={len(unique_drugs)} | proteins={len(unique_targets)}"
    )
    return {
        "ligand_embeddings": ligand_embeddings,
        "ligand_graphs": ligand_graphs,
        "protein_store": protein_store,
        "protein_graph_store": protein_graph_store,
        "lengths_lookup": lengths_lookup,
        "available_ligand_ids": set(unique_drugs),
        "available_target_ids": set(unique_targets),
    }

def run_single_seed_scene(seed: int, scene: str, args, runtime_bundle: Optional[dict] = None) -> dict:
    print("=" * 110)
    print(f"[{args.dataset_name}][Seed {seed}][Scene {scene}] Starting")
    set_global_seed(seed, deterministic=args.deterministic)
    scene_cfg = get_scene_runtime_config(scene, args)
    update_global_progress(
        args.model_root,
        {
            "dataset": args.dataset_name,
            "state": "running",
            "seed": int(seed),
            "scene": scene,
            "scene_cfg": scene_cfg,
            "timestamp": time.time(),
        },
    )

    split_paths = get_scene_split_paths(args.splits_dir, seed, scene)
    split_md5 = md5_files([split_paths["train"], split_paths["val"], split_paths["test"]])

    scene_dir = ensure_dir(os.path.join(args.model_root, f"seed_{seed:04d}", scene))
    config_path = os.path.join(scene_dir, "config.json")
    status_path = os.path.join(scene_dir, "status.json")
    train_log_path = os.path.join(scene_dir, "train_log.csv")
    best_ckpt_path = os.path.join(scene_dir, "best.pt")
    last_ckpt_path = os.path.join(scene_dir, "last.pt")

    df = prepare_scene_split_dataframe(split_paths)
    if runtime_bundle is None:
        runtime_bundle = prepare_dataset_runtime(args)
    valid_lig_ids = runtime_bundle.get("available_ligand_ids", set())
    valid_target_ids = runtime_bundle.get("available_target_ids", set())
    missing_drugs = sorted(set(df["drug_id"]) - set(valid_lig_ids))
    missing_targets = sorted(set(df["target_id"]) - set(valid_target_ids))
    if missing_drugs or missing_targets:
        raise FileNotFoundError(
            f"[{args.dataset_name}][Seed {seed}][Scene {scene}] selected split refers to entities "
            f"without valid precomputed artifacts: drugs={missing_drugs[:10]}, "
            f"targets={missing_targets[:10]}. Rerun embedding_generation.py."
        )
    split_counts = summarize_split_counts(df)
    print(f"[{args.dataset_name}][Seed {seed}][Scene {scene}] Split counts: {split_counts}")

    train_df = df[df["Split"] == "train"].reset_index(drop=True)
    val_df = df[df["Split"] == "val"].reset_index(drop=True)
    test_df = df[df["Split"] == "test"].reset_index(drop=True)

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError(
            f"[{args.dataset_name}][Seed {seed}][Scene {scene}] train/val/test contains an empty split: {split_counts}"
        )

    all_ligand_ids = [normalize_entity_id(x) for x in df["drug_id"].tolist()]
    all_seq_hashes = [normalize_entity_id(x) for x in df["target_id"].tolist()]

    lengths_lookup = runtime_bundle["lengths_lookup"]
    ligand_embeddings = runtime_bundle["ligand_embeddings"]
    ligand_graphs = runtime_bundle["ligand_graphs"]
    protein_store = runtime_bundle["protein_store"]
    protein_graph_store = runtime_bundle["protein_graph_store"]

    train_mean = float(train_df["affinity"].mean())
    train_std = float(train_df["affinity"].std(ddof=0))
    if train_std < 1e-8:
        train_std = 1.0

    train_loader = build_dataloader(
        df=train_df,
        ligand_embeddings=ligand_embeddings,
        ligand_graphs=ligand_graphs,
        protein_store=protein_store,
        protein_graph_store=protein_graph_store,
        train_mean=train_mean,
        train_std=train_std,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        max_protein_len=args.max_protein_len,
        lengths_lookup=lengths_lookup,
        shuffle=True,
    )
    val_loader = build_dataloader(
        df=val_df,
        ligand_embeddings=ligand_embeddings,
        ligand_graphs=ligand_graphs,
        protein_store=protein_store,
        protein_graph_store=protein_graph_store,
        train_mean=train_mean,
        train_std=train_std,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        max_protein_len=args.max_protein_len,
        lengths_lookup=lengths_lookup,
        shuffle=False,
    )
    test_loader = build_dataloader(
        df=test_df,
        ligand_embeddings=ligand_embeddings,
        ligand_graphs=ligand_graphs,
        protein_store=protein_store,
        protein_graph_store=protein_graph_store,
        train_mean=train_mean,
        train_std=train_std,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        max_protein_len=args.max_protein_len,
        lengths_lookup=lengths_lookup,
        shuffle=False,
    )

    device = torch.device(args.device)
    model = UEAlignNet12A(
        ligand_dim=args.ligand_dim,
        graph_in_dim=DEFAULT_ATOM_FEAT_DIM,
        protein_dim=args.protein_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_graph_layers=args.num_graph_layers,
        num_protein_graph_layers=args.num_protein_graph_layers,
        num_sparse_interaction_layers=args.num_sparse_interaction_layers,
        num_attention_heads=args.num_attention_heads,
        residue_topk=args.residue_selector_topk,
        use_ligand_global=not args.disable_ligand_global_shortcut,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=scene_cfg["lr"],
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-6,
    )
    total_steps = max(1, len(train_loader) * scene_cfg["max_epochs"])
    warmup_steps = len(train_loader) * scene_cfg["warmup_epochs"]
    scheduler = build_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and device.type == "cuda")) if device.type == "cuda" else None
    loss_fn = RegressionRankingLoss(huber_beta=args.huber_beta, rank_weight=scene_cfg.get("rank_loss_weight", args.rank_loss_weight))

    start_epoch = 0
    best_val_rmse = float("inf")
    best_val_spearman = -1.0
    best_val_score = float("-inf")
    patience_counter = 0
    completed = False

    status = load_json(status_path, default={}) or {}
    if os.path.exists(config_path):
        old_config = load_json(config_path, default={}) or {}
        old_split_md5 = old_config.get("split_md5")
        if old_split_md5 and old_split_md5 != split_md5:
            raise RuntimeError(
                f"[{args.dataset_name}][Seed {seed}][Scene {scene}] Split file md5 changed. "
                f"Old={old_split_md5}, New={split_md5}. Refusing to mix checkpoints."
            )

    resume_mode, resume_ckpt = choose_resume_checkpoint(
        best_ckpt_path=best_ckpt_path,
        last_ckpt_path=last_ckpt_path,
        status=status,
        device=device,
        split_md5=split_md5,
    )
    if resume_mode == "completed_best":
        print(f"[{args.dataset_name}][Seed {seed}][Scene {scene}] Found completed status + best checkpoint. Skip training.")
        ckpt = resume_ckpt
        model.load_state_dict(ckpt["model_state"])
        train_mean = float(ckpt.get("train_mean", train_mean))
        train_std = float(ckpt.get("train_std", train_std))
        best_val_rmse = float(ckpt.get("best_val_rmse", best_val_rmse))
        best_val_spearman = float(ckpt.get("best_val_spearman", best_val_spearman))
        best_val_score = float(ckpt.get("best_val_score", best_val_score))
        completed = True
        if args.skip_eval_if_completed:
            metrics_val_path = os.path.join(scene_dir, "metrics_val.json")
            metrics_test_path = os.path.join(scene_dir, "metrics_test.json")
            all_metrics_path = os.path.join(scene_dir, "all_metrics.json")
            if os.path.exists(metrics_val_path) and os.path.exists(metrics_test_path) and os.path.exists(all_metrics_path):
                print(f"[{args.dataset_name}][Seed {seed}][Scene {scene}] Existing evaluation results found; skipping duplicate val/test evaluation.")
                data = load_json(all_metrics_path, default=None)
                if isinstance(data, dict) and "test" in data and "val" in data:
                    update_global_progress(
                        args.model_root,
                        {
                            "dataset": args.dataset_name,
                            "state": "scene_finished",
                            "seed": int(seed),
                            "scene": scene,
                            "test_metrics": data.get("test", {}),
                            "timestamp": time.time(),
                        },
                    )
                    return data
    elif resume_mode in {"resume_last", "resume_best_only"}:
        resume_msg = "last.pt" if resume_mode == "resume_last" else "best.pt (last.pt missing or older)"
        print(f"[{args.dataset_name}][Seed {seed}][Scene {scene}] Resuming from {resume_msg}.")
        ckpt = resume_ckpt
        model.load_state_dict(ckpt["model_state"])
        if ckpt.get("optimizer_state") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scheduler_state") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if ckpt.get("scaler_state") is not None and scaler is not None and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_val_rmse = float(ckpt.get("best_val_rmse", float("inf")))
        best_val_spearman = float(ckpt.get("best_val_spearman", -1.0))
        best_val_score = float(ckpt.get("best_val_score", float("-inf")))
        patience_counter = int(ckpt.get("patience_counter", 0))
        train_mean = float(ckpt.get("train_mean", train_mean))
        train_std = float(ckpt.get("train_std", train_std))

    config_to_save = vars(args).copy()
    config_to_save["seed"] = seed
    config_to_save["scene"] = scene
    config_to_save["scene_split_dir"] = split_paths["scene_dir"]
    config_to_save["split_md5"] = split_md5
    config_to_save["split_counts"] = split_counts
    config_to_save["atom_feat_dim"] = DEFAULT_ATOM_FEAT_DIM
    config_to_save["scene_runtime_config"] = scene_cfg
    save_json(config_to_save, config_path)

    if not completed:
        print(f"[{args.dataset_name}][Seed {seed}][Scene {scene}] Training from epoch {start_epoch} to {scene_cfg['max_epochs'] - 1} with lr={scene_cfg['lr']:.2e}, patience={scene_cfg['patience']}")
        if not os.path.exists(train_log_path):
            with open(train_log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "epoch", "train_loss", "train_reg_loss", "train_rank_loss",
                        "skipped_batches", "val_rmse", "val_mse", "val_r2", "val_ci", "val_pearson",
                        "val_spearman", "val_ckpt_score", "lr", "seconds",
                    ]
                )

        for epoch in range(start_epoch, scene_cfg["max_epochs"]):
            epoch_start = time.time()
            train_stats = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                loss_fn=loss_fn,
                device=device,
                grad_clip=args.grad_clip,
                use_amp=args.amp,
                desc=f"{args.dataset_name} Seed {seed} {scene} Epoch {epoch} [train]",
            )

            _, val_metrics = predict_epoch(
                model=model,
                loader=val_loader,
                device=device,
                train_mean=train_mean,
                train_std=train_std,
                use_amp=args.amp,
                desc=f"{args.dataset_name} Seed {seed} {scene} Epoch {epoch} [val]",
            )
            val_ckpt_score = compute_checkpoint_score(val_metrics, spearman_weight=args.ckpt_spearman_weight)
            epoch_seconds = time.time() - epoch_start
            current_lr = float(optimizer.param_groups[0]["lr"])

            print(
                f"[{args.dataset_name}][Seed {seed}][Scene {scene}][Epoch {epoch}] "
                f"train_loss={train_stats['loss']:.5f} "
                f"(reg={train_stats['reg_loss']:.5f}, rank={train_stats['rank_loss']:.5f}, "
                f"skipped={train_stats.get('skipped_batches', 0)}) "
                f"val: {format_metrics(val_metrics)} "
                f"ckpt_score={val_ckpt_score:.5f} "
                f"lr={current_lr:.6e} "
                f"time={epoch_seconds:.1f}s"
            )

            with open(train_log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        epoch,
                        train_stats["loss"],
                        train_stats["reg_loss"],
                        train_stats["rank_loss"],
                        train_stats.get("skipped_batches", 0),
                        val_metrics["rmse"],
                        val_metrics["mse"],
                        val_metrics["r2"],
                        val_metrics["ci"],
                        val_metrics["pearson"],
                        val_metrics["spearman"],
                        val_ckpt_score,
                        current_lr,
                        epoch_seconds,
                    ]
                )

            is_best = should_save_best(
                val_metrics,
                best_val_rmse=best_val_rmse,
                best_val_spearman=best_val_spearman,
                rmse_min_delta=args.min_delta,
                rmse_tolerance=scene_cfg["ckpt_rmse_tolerance"],
                spearman_min_delta=args.ckpt_spearman_min_delta,
            )
            if is_best:
                best_val_rmse = float(val_metrics["rmse"])
                best_val_spearman = float(val_metrics.get("spearman", float("nan")))
                if not np.isfinite(best_val_spearman):
                    best_val_spearman = -1.0
                best_val_score = float(val_ckpt_score)
                patience_counter = 0
                save_checkpoint(
                    path=best_ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler if (scaler is not None and scaler.is_enabled()) else None,
                    epoch=epoch,
                    best_val_rmse=best_val_rmse,
                    best_val_spearman=best_val_spearman,
                    best_val_score=best_val_score,
                    patience_counter=patience_counter,
                    train_mean=train_mean,
                    train_std=train_std,
                    split_md5=split_md5,
                    seed=seed,
                    scene=scene,
                    config=config_to_save,
                )
            else:
                patience_counter += 1

            save_checkpoint(
                path=last_ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler if (scaler is not None and scaler.is_enabled()) else None,
                epoch=epoch,
                best_val_rmse=best_val_rmse,
                best_val_spearman=best_val_spearman,
                best_val_score=best_val_score,
                patience_counter=patience_counter,
                train_mean=train_mean,
                train_std=train_std,
                split_md5=split_md5,
                seed=seed,
                scene=scene,
                config=config_to_save,
            )

            status = {
                "seed": seed,
                "scene": scene,
                "dataset": args.dataset_name,
                "scene_split_dir": split_paths["scene_dir"],
                "split_md5": split_md5,
                "epoch_finished": epoch,
                "best_val_rmse": best_val_rmse,
                "best_val_spearman": best_val_spearman,
                "best_val_score": best_val_score,
                "patience_counter": patience_counter,
                "completed": False,
            }
            save_json(status, status_path)
            update_global_progress(
                args.model_root,
                {
                    "dataset": args.dataset_name,
                    "state": "running",
                    "seed": int(seed),
                    "scene": scene,
                    "epoch_finished": int(epoch),
                    "best_val_rmse": float(best_val_rmse),
                    "best_val_spearman": float(best_val_spearman),
                    "timestamp": time.time(),
                },
            )

            if patience_counter >= scene_cfg["patience"]:
                print(f"[{args.dataset_name}][Seed {seed}][Scene {scene}] Early stopped at epoch {epoch}.")
                break

        status = {
            "seed": seed,
            "scene": scene,
            "dataset": args.dataset_name,
            "scene_split_dir": split_paths["scene_dir"],
            "split_md5": split_md5,
            "best_val_rmse": best_val_rmse,
            "best_val_spearman": best_val_spearman,
            "best_val_score": best_val_score,
            "completed": True,
        }
        save_json(status, status_path)
        update_global_progress(
            args.model_root,
            {
                "dataset": args.dataset_name,
                "state": "scene_completed",
                "seed": int(seed),
                "scene": scene,
                "best_val_rmse": float(best_val_rmse),
                "best_val_spearman": float(best_val_spearman),
                "timestamp": time.time(),
            },
        )

        if not os.path.exists(best_ckpt_path):
            raise RuntimeError(f"[{args.dataset_name}][Seed {seed}][Scene {scene}] Training ended but best.pt was not created.")

        ckpt = load_checkpoint(best_ckpt_path, device=device)
        model.load_state_dict(ckpt["model_state"])
        train_mean = float(ckpt.get("train_mean", train_mean))
        train_std = float(ckpt.get("train_std", train_std))
    else:
        if not os.path.exists(best_ckpt_path):
            raise RuntimeError(f"[{args.dataset_name}][Seed {seed}][Scene {scene}] completed=True but best.pt is missing.")
        ckpt = load_checkpoint(best_ckpt_path, device=device)
        model.load_state_dict(ckpt["model_state"])
        train_mean = float(ckpt.get("train_mean", train_mean))
        train_std = float(ckpt.get("train_std", train_std))

    val_metrics = evaluate_and_save(model, "val", val_loader, device, train_mean, train_std, scene_dir, use_amp=args.amp)
    test_metrics = evaluate_and_save(model, "test", test_loader, device, train_mean, train_std, scene_dir, use_amp=args.amp)

    all_metrics = {
        "dataset": args.dataset_name,
        "seed": seed,
        "scene": scene,
        "val": val_metrics,
        "test": test_metrics,
        "best_val_rmse": best_val_rmse,
        "best_val_spearman": best_val_spearman,
        "best_val_score": best_val_score,
    }
    save_json(all_metrics, os.path.join(scene_dir, "all_metrics.json"))
    print(f"[{args.dataset_name}][Seed {seed}][Scene {scene}] Finished.")
    update_global_progress(
        args.model_root,
        {
            "dataset": args.dataset_name,
            "state": "scene_finished",
            "seed": int(seed),
            "scene": scene,
            "test_metrics": test_metrics,
            "timestamp": time.time(),
        },
    )
    return all_metrics

def aggregate_results(model_root: str, seeds: Sequence[int], target_scenarios: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for seed in seeds:
        for scene in target_scenarios:
            metric_file = os.path.join(model_root, f"seed_{seed:04d}", scene, "all_metrics.json")
            if not os.path.exists(metric_file):
                continue
            data = load_json(metric_file, default=None)
            if data is None or "test" not in data:
                continue
            row = {"seed": seed, "scene": scene}
            row.update(data["test"])
            rows.append(row)

    detail_df = pd.DataFrame(rows)
    if detail_df.empty:
        return detail_df, pd.DataFrame()

    summary_df = build_standard_summary(detail_df)
    detail_path = os.path.join(model_root, "all_seed_scene_test_metrics.csv")
    summary_path = os.path.join(model_root, "summary_mean_std_by_scene.csv")
    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False, float_format="%.4f")
    save_json(
        {
            "detail_csv": detail_path,
            "summary_csv": summary_path,
            "n_seeds_collected": int(detail_df["seed"].nunique()),
            "scenes": list(target_scenarios),
        },
        os.path.join(model_root, "summary_manifest.json"),
    )
    return detail_df, summary_df

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train UE-AlignNet on standardized Neuroinflam-VSF Davis, KIBA, "
            "and BindingDB splits using precomputed feature manifests."
        )
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="davis / kiba / bindingdb / all. The alias both runs Davis and KIBA.",
    )
    parser.add_argument("--data_root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--embedding_root", type=str, default=str(DEFAULT_EMBEDDING_ROOT))
    parser.add_argument("--model_base", type=str, default=str(DEFAULT_MODEL_BASE))
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--splits_dir",
        type=str,
        default=None,
        help="Optional split root override for a single selected dataset.",
    )

    parser.add_argument("--seeds", type=str, default="1-5")
    parser.add_argument(
        "--target_scenarios",
        type=str,
        default=",".join(DEFAULT_TARGET_SCENARIOS),
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--non_deterministic", dest="deterministic", action="store_false")

    # Model architecture.
    parser.add_argument("--ligand_dim", type=int, default=DEFAULT_LIGAND_DIM)
    parser.add_argument("--protein_dim", type=int, default=DEFAULT_PROTEIN_DIM)
    parser.add_argument("--hidden_dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--num_graph_layers", type=int, default=3)
    parser.add_argument("--num_protein_graph_layers", type=int, default=2)
    parser.add_argument("--num_sparse_interaction_layers", type=int, default=2)
    parser.add_argument("--residue_selector_topk", type=int, default=128)
    parser.add_argument("--disable_ligand_global_shortcut", action="store_true", default=False)
    parser.add_argument("--num_attention_heads", type=int, default=8)

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true", default=True)
    parser.add_argument("--no_pin_memory", dest="pin_memory", action="store_false")
    parser.add_argument("--max_protein_len", type=int, default=None, help="0 means no downsampling")
    parser.add_argument("--max_ligand_atoms", type=int, default=0, help="0 means no atom truncation")
    parser.add_argument("--protein_cache_size", type=int, default=20000)
    parser.add_argument("--protein_graph_cache_size", type=int, default=20000)
    parser.add_argument(
        "--preload_all_data_to_memory",
        dest="preload_all_data_to_memory",
        action="store_true",
        help="Load already-generated ligand/protein artifacts into RAM before training.",
    )
    parser.add_argument(
        "--no_preload_all_data_to_memory",
        dest="preload_all_data_to_memory",
        action="store_false",
    )
    parser.set_defaults(preload_all_data_to_memory=None)
    parser.add_argument(
        "--allow_full_length_memory_preload", action="store_true", default=False
    )
    parser.add_argument(
        "--preload_proteins", action="store_true", default=False,
        help="Preload ESMC tensors into the disk-backed LRU store when full RAM preload is disabled.",
    )
    parser.add_argument(
        "--protein_graph_mode",
        type=str,
        default="contact",
        choices=["contact", "contact_plus_band", "hybrid"],
    )
    parser.add_argument("--protein_seq_band_width", type=int, default=2)
    parser.add_argument("--protein_contact_threshold", type=float, default=0.0)

    # Training and early-stopping settings.
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--scene_aware_training", action="store_true", default=True)
    parser.add_argument("--no_scene_aware_training", dest="scene_aware_training", action="store_false")
    parser.add_argument("--cold_target_lr", type=float, default=2e-4)
    parser.add_argument("--cold_target_max_epochs", type=int, default=None)
    parser.add_argument("--cold_target_patience", type=int, default=None)
    parser.add_argument("--cold_target_warmup_epochs", type=int, default=6)
    parser.add_argument("--cold_target_ckpt_rmse_tolerance", type=float, default=0.006)
    parser.add_argument("--double_cold_lr", type=float, default=2e-4)
    parser.add_argument("--double_cold_max_epochs", type=int, default=None)
    parser.add_argument("--double_cold_patience", type=int, default=None)
    parser.add_argument("--double_cold_warmup_epochs", type=int, default=6)
    parser.add_argument("--double_cold_ckpt_rmse_tolerance", type=float, default=0.010)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--ckpt_rmse_tolerance", type=float, default=0.003)
    parser.add_argument("--ckpt_spearman_min_delta", type=float, default=5e-4)
    parser.add_argument("--ckpt_spearman_weight", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--huber_beta", type=float, default=1.0)
    parser.add_argument("--rank_loss_weight", type=float, default=0.2)
    parser.add_argument("--cold_target_rank_loss_weight", type=float, default=0.05)
    parser.add_argument("--double_cold_rank_loss_weight", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument(
        "--skip_eval_if_completed", action="store_true", default=True,
        help="If a completed task already has val/test metrics, reuse them.",
    )
    parser.add_argument(
        "--force_eval_completed", dest="skip_eval_if_completed", action="store_false",
        help="Force val/test evaluation even if a completed task already has metrics.",
    )
    return parser


def normalize_dataset_choice(value: str) -> str:
    choice = str(value).strip().lower()
    mapping = {
        "1": "davis",
        "davis": "davis",
        "2": "kiba",
        "kiba": "kiba",
        "3": "bindingdb",
        "bindingdb": "bindingdb",
        "binding": "bindingdb",
        "4": "all",
        "all": "all",
        "three": "all",
        "both": "both",
    }
    if choice not in mapping:
        raise ValueError(
            "Invalid dataset choice. Use davis, kiba, bindingdb, all, or both."
        )
    return mapping[choice]


def choose_dataset_interactively() -> str:
    print("\nSelect a dataset:")
    print("1. Train and test on Davis")
    print("2. Train and test on KIBA")
    print("3. Train and test on BindingDB")
    print("4. Train and test on all three DTA datasets")
    return normalize_dataset_choice(input("Enter 1 / 2 / 3 / 4: ").strip())


def dataset_display_name(dataset_key: str) -> str:
    mapping = {
        "davis": "Davis",
        "kiba": "KIBA",
        "bindingdb": "BindingDB",
    }
    key = str(dataset_key).strip().lower()
    if key not in mapping:
        raise ValueError(f"Unsupported DTA dataset: {dataset_key}")
    return mapping[key]


def apply_dataset_runtime_defaults(args) -> None:
    defaults = DATASET_RUNTIME_DEFAULTS[args.dataset_name]
    for name, value in defaults.items():
        if getattr(args, name, None) is None:
            setattr(args, name, value)


def configure_dataset_args(base_args, dataset_key: str):
    args = argparse.Namespace(**vars(base_args))
    args.dataset_key = dataset_key.lower()
    args.dataset_name = dataset_display_name(args.dataset_key)
    args.data_root = os.path.abspath(os.path.expanduser(str(args.data_root)))
    args.embedding_root = os.path.abspath(os.path.expanduser(str(args.embedding_root)))
    args.model_base = os.path.abspath(os.path.expanduser(str(args.model_base)))

    if args.splits_dir:
        args.splits_dir = os.path.abspath(os.path.expanduser(str(args.splits_dir)))
    else:
        args.splits_dir = os.path.join(args.data_root, args.dataset_name, "splits")

    args.model_root = os.path.join(
        args.model_base, args.dataset_name, args.model_name
    )
    apply_dataset_runtime_defaults(args)

    if args.max_protein_len is not None and int(args.max_protein_len) <= 0:
        args.max_protein_len = None
    if args.max_ligand_atoms is not None and int(args.max_ligand_atoms) <= 0:
        args.max_ligand_atoms = None
    return args


def run_dataset(args) -> None:
    ensure_dir(args.model_root)
    ensure_scene_splits_exist(args)

    print("\n" + "#" * 110)
    print(f"🚀 {args.model_name} on {args.dataset_name}")
    print(f"Split root      : {args.splits_dir}")
    print(f"Embedding root  : {args.embedding_root}")
    print(f"Model root      : {args.model_root}")
    print(f"Seeds           : {args.seeds}")
    print(f"Scenarios       : {args.target_scenarios}")
    print("Feature policy  : READ-ONLY; missing artifacts cause an error")
    print("#" * 110)

    runtime_bundle = prepare_dataset_runtime(args)
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    all_results = []
    for seed in args.seeds:
        print("\n" + "#" * 110)
        print(f"########## {args.dataset_name}: Start all target scenes for seed {seed} ##########")
        print("#" * 110)
        for scene in args.target_scenarios:
            all_results.append(
                run_single_seed_scene(seed, scene, args, runtime_bundle=runtime_bundle)
            )

    detail_df, summary_df = aggregate_results(
        args.model_root, args.seeds, args.target_scenarios
    )
    update_global_progress(
        args.model_root,
        {"dataset": args.dataset_name, "state": "all_done", "timestamp": time.time()},
    )
    if not detail_df.empty:
        print("=" * 110)
        print(f"[{args.dataset_name}][Summary] Per-seed metrics:")
        print(os.path.join(args.model_root, "all_seed_scene_test_metrics.csv"))
        print(f"[{args.dataset_name}][Summary] Mean ± std:")
        print(format_standard_summary(summary_df))


def main():
    parser = build_argparser()
    base_args = parser.parse_args()
    dataset_choice = (
        normalize_dataset_choice(base_args.dataset)
        if base_args.dataset is not None
        else choose_dataset_interactively()
    )
    base_args.seeds = parse_seed_list(base_args.seeds)
    base_args.target_scenarios = parse_scenario_list(base_args.target_scenarios)

    if dataset_choice == "all":
        datasets = ["davis", "kiba", "bindingdb"]
    elif dataset_choice == "both":
        datasets = ["davis", "kiba"]
    else:
        datasets = [dataset_choice]

    if base_args.splits_dir and len(datasets) != 1:
        raise ValueError("--splits_dir can only be used when a single dataset is selected.")

    for dataset_key in datasets:
        args = configure_dataset_args(base_args, dataset_key)
        run_dataset(args)


if __name__ == "__main__":
    main()
