#!/usr/bin/env python3
"""Screen ChEMBL compounds for BBB permeability and predicted NLRP3 affinity.

The default workflow downloads ChEMBL phase-4 small molecules, standardizes
parent structures, applies the BP-NET seed ensemble, and ranks BBB-permeable
compounds with BindingDB cold-target UE-AlignNet checkpoints. Validated feature
caches and prediction chunks support resumable execution.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable: Iterable[Any], **_: Any) -> Iterable[Any]:
        return iterable


SCRIPT_VERSION = "ChEMBL_BPNet_UEAlignNet_NLRP3_screen_v2"
NLRP3_ACCESSION = "Q96P20"
NLRP3_ENTITY_ID = "Q96P20-1"
NLRP3_EXPECTED_LENGTH = 1036
NLRP3_FASTA_URL = (
    f"https://rest.uniprot.org/uniprotkb/{NLRP3_ACCESSION}.fasta"
)
ALLOWED_SCENARIOS = ("warm", "cold_drug", "cold_target", "double_cold")

DEFAULT_CHEMBL_ROOT = Path("./data/datasets/virtual_screening/ChEMBL")
DEFAULT_CHEMBL_SMALL_MOLECULE_CSV = DEFAULT_CHEMBL_ROOT / (
    "chembl_max_phase_4_small_molecule_only.csv"
)
DEFAULT_BBB_OUTPUT_DIR = DEFAULT_CHEMBL_ROOT / "BP-NET"
DEFAULT_BBB_CANDIDATE_CSV = DEFAULT_BBB_OUTPUT_DIR / (
    "chembl_max_phase_4_small_molecule_only_"
    "parent_deduplicated_BBB_permeable_primary.csv"
)
DEFAULT_NLRP3_FASTA = Path("./data/datasets/BindingDB/NLRP3_Q96P20.fasta")
DEFAULT_EMBEDDING_ROOT = Path("./data/embedding")
DEFAULT_BP_FEATURE_CACHE_ROOT = Path(
    "./data/embedding/virtual_screening/NLRP3/BP-NET"
)
DEFAULT_UE_FEATURE_CACHE_ROOT = Path(
    "./data/embedding/virtual_screening/NLRP3/UE-AlignNet"
)
DEFAULT_BP_MODEL_ROOT = Path("./models/BP-NET")
DEFAULT_UE_MODEL_ROOT = Path("./models/UE-AlignNet")
DEFAULT_DATA_ROOT = Path("./data/datasets")
DEFAULT_OUTPUT_DIR = Path(
    "./data/datasets/virtual_screening/results/NLRP3_UE-AlignNet"
)

EXTERNAL_METADATA_EXACT_COLUMNS = {
    "candidate_origin",
    "is_external_control",
    "is_formal_chembl_structure",
    "control_overlaps_formal_chembl",
    "control_label_binary",
    "is_known_positive_control",
    "is_expected_positive_pair",
    "experimental_affinity_type",
    "experimental_affinity_text_nm",
    "experimental_affinity_relation",
    "experimental_affinity_nm",
    "experimental_affinity_grade",
    "experimental_affinity_all_records",
}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(json_safe(value), stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def torch_load(path: Path, map_location: Any = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # Compatibility fallback when weights_only is unsupported.
        return torch.load(path, map_location=map_location)


def import_project_module(name: str, path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Missing project module: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import project module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CHEMBL_API_URL = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"


def clean_optional_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "null", "<na>"} else text


def nested_value(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def load_raw_chembl_cache(raw_path: Path, meta_path: Path) -> list[dict[str, Any]] | None:
    if not raw_path.is_file() or not meta_path.is_file():
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as stream:
            meta = json.load(stream)
        if not meta.get("download_complete") or meta.get("max_phase") != 4:
            return None
        records: list[dict[str, Any]] = []
        with raw_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        return None
                    records.append(record)
        if len(records) != int(meta.get("record_count", -1)):
            return None
        return records
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def download_chembl_max_phase_4(
    raw_path: Path,
    meta_path: Path,
    page_size: int,
    retry_count: int,
) -> list[dict[str, Any]]:
    try:
        import requests  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("ChEMBL download requires requests") from exc

    records: list[dict[str, Any]] = []
    offset = 0
    expected_total: int | None = None
    progress = tqdm(desc="ChEMBL max_phase=4", unit="molecule")
    try:
        while True:
            payload = None
            last_error: Exception | None = None
            for attempt in range(1, retry_count + 1):
                try:
                    response = requests.get(
                        CHEMBL_API_URL,
                        params={"max_phase": 4, "limit": page_size, "offset": offset},
                        timeout=60,
                    )
                    response.raise_for_status()
                    payload = response.json()
                    break
                except Exception as exc:  # Retry transient network and server failures.
                    last_error = exc
                    if attempt < retry_count:
                        time.sleep(min(2 * attempt, 10))
            if payload is None:
                raise RuntimeError(
                    f"ChEMBL download failed at offset={offset}: {last_error}"
                )
            page = payload.get("molecules") or []
            page_meta = payload.get("page_meta") or {}
            if expected_total is None and page_meta.get("total_count") is not None:
                expected_total = int(page_meta["total_count"])
                progress.total = expected_total
            records.extend(record for record in page if isinstance(record, dict))
            progress.update(len(page))
            if not page or not page_meta.get("next"):
                break
            offset += page_size
    finally:
        progress.close()

    if expected_total is not None and len(records) != expected_total:
        raise RuntimeError(
            f"Incomplete ChEMBL download: received={len(records)}, expected={expected_total}"
        )
    bad_phase = [record.get("max_phase") for record in records if record.get("max_phase") != 4]
    if bad_phase:
        raise RuntimeError(f"ChEMBL API returned records outside max_phase=4: {bad_phase[:5]}")

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = raw_path.parent / f".{raw_path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, raw_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    atomic_write_json(meta_path, {
        "download_complete": True,
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "api_url": CHEMBL_API_URL,
        "max_phase": 4,
        "record_count": len(records),
    })
    return records


def normalize_chembl_records(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    lookup = {
        clean_optional_text(record.get("molecule_chembl_id")): record
        for record in records
        if clean_optional_text(record.get("molecule_chembl_id"))
    }
    rows: list[dict[str, Any]] = []
    for record in records:
        structures = record.get("molecule_structures") or {}
        properties = record.get("molecule_properties") or {}
        hierarchy = record.get("molecule_hierarchy") or {}
        parent_id = clean_optional_text(hierarchy.get("parent_chembl_id"))
        parent = lookup.get(parent_id, {})
        parent_structures = parent.get("molecule_structures") or {}
        rows.append({
            "chembl_id": clean_optional_text(record.get("molecule_chembl_id")),
            "pref_name": clean_optional_text(record.get("pref_name")),
            "max_phase": record.get("max_phase"),
            "molecule_type": clean_optional_text(record.get("molecule_type")),
            "canonical_smiles": clean_optional_text(structures.get("canonical_smiles")),
            "standard_inchi": clean_optional_text(structures.get("standard_inchi")),
            "standard_inchi_key": clean_optional_text(structures.get("standard_inchi_key")),
            "first_approval": record.get("first_approval"),
            "oral": record.get("oral"),
            "parenteral": record.get("parenteral"),
            "topical": record.get("topical"),
            "black_box_warning": record.get("black_box_warning"),
            "withdrawn_flag": record.get("withdrawn_flag"),
            "prodrug": record.get("prodrug"),
            "structure_type": record.get("structure_type"),
            "full_mwt": properties.get("full_mwt"),
            "mw_freebase": properties.get("mw_freebase"),
            "alogp": properties.get("alogp"),
            "psa": properties.get("psa"),
            "hba": properties.get("hba"),
            "hbd": properties.get("hbd"),
            "rtb": properties.get("rtb"),
            "num_ro5_violations": properties.get("num_ro5_violations"),
            "hierarchy_parent_chembl_id": parent_id,
            "hierarchy_parent_pref_name": clean_optional_text(parent.get("pref_name")),
            "hierarchy_parent_canonical_smiles": clean_optional_text(
                parent_structures.get("canonical_smiles")
            ),
            "hierarchy_active_chembl_id": clean_optional_text(
                hierarchy.get("active_chembl_id")
            ),
        })
    return pd.DataFrame(rows)


def prepare_chembl_small_molecules(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    output = args.chembl_small_molecule_csv
    audit_path = output.with_name(output.stem + "_audit.csv")
    summary_path = output.with_name(output.stem + "_summary.json")
    if (
        args.chembl_source_csv is None
        and output.is_file()
        and output.stat().st_size > 0
        and not args.force_download
    ):
        frame = pd.read_csv(output, low_memory=False)
        if {"chembl_id", "max_phase", "molecule_type", "canonical_smiles"}.issubset(frame):
            phases = pd.to_numeric(frame["max_phase"], errors="coerce")
            types = frame["molecule_type"].fillna("").astype(str).str.casefold()
            if len(frame) > 0 and bool(phases.eq(4).all()) and bool(types.eq("small molecule").all()):
                return output, {
                    "source": "validated_processed_cache",
                    "small_molecule_records": len(frame),
                    "output_csv": str(output),
                }

    if args.chembl_source_csv is not None:
        source = args.chembl_source_csv
        if not source.is_file() or source.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing/empty --chembl-source-csv: {source}")
        frame = pd.read_csv(source, low_memory=False)
        source_label = str(source)
    else:
        raw_path = args.chembl_root / "raw" / "chembl_max_phase_4.jsonl"
        meta_path = raw_path.with_suffix(".meta.json")
        records = None if args.force_download else load_raw_chembl_cache(raw_path, meta_path)
        if records is None:
            if args.check_only:
                raise FileNotFoundError(
                    f"ChEMBL cache is absent: {raw_path}. Run without --check-only first."
                )
            records = download_chembl_max_phase_4(
                raw_path, meta_path, args.chembl_page_size, args.download_retries
            )
            source_label = CHEMBL_API_URL
        else:
            source_label = str(raw_path)
        frame = normalize_chembl_records(records)

    required = {"max_phase", "molecule_type", "canonical_smiles"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"ChEMBL input is missing columns: {sorted(missing)}")
    phase = pd.to_numeric(frame["max_phase"], errors="coerce")
    molecule_type = frame["molecule_type"].fillna("").astype(str).str.strip().str.casefold()
    keep = phase.eq(4) & molecule_type.eq("small molecule")
    audit = frame.copy()
    audit["small_molecule_keep"] = keep
    audit["upstream_exclusion_reason"] = np.where(
        ~phase.eq(4), "not_exact_max_phase_4",
        np.where(~molecule_type.eq("small molecule"), "non_small_molecule", ""),
    )
    selected = frame.loc[keep].copy().reset_index(drop=True)
    if selected.empty:
        raise RuntimeError("No exact max_phase=4 Small molecule records remain")
    atomic_write_csv(audit, audit_path)
    atomic_write_csv(selected, output)
    summary = {
        "source": source_label,
        "source_records": len(frame),
        "small_molecule_records": len(selected),
        "excluded_records": int((~keep).sum()),
        "scientific_filter": "exact max_phase=4 and molecule_type=Small molecule only",
        "output_csv": str(output),
        "audit_csv": str(audit_path),
    }
    atomic_write_json(summary_path, summary)
    return output, summary


def standardize_bbb_structure(smiles: str, hierarchy_parent_smiles: str) -> dict[str, Any]:
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    source = clean_optional_text(smiles)
    parent = clean_optional_text(hierarchy_parent_smiles)
    selected = parent or source
    method = "chembl_hierarchy_parent" if parent else "original_structure"
    molecule = Chem.MolFromSmiles(selected)
    if molecule is None:
        raise ValueError("RDKit cannot parse selected structure")
    if len(Chem.GetMolFrags(molecule)) > 1:
        fragments = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True)
        if not fragments:
            raise ValueError("No valid fragments after multicomponent parsing")
        molecule = max(
            fragments,
            key=lambda item: (item.GetNumHeavyAtoms(), item.GetNumAtoms()),
        )
        method += "+largest_parent_fragment"
    try:
        molecule = rdMolStandardize.Uncharger().uncharge(molecule)
        Chem.SanitizeMol(molecule)
        method += "+uncharged"
    except Exception:
        Chem.SanitizeMol(molecule)
        method += "+charge_retained"
    model_smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    if not model_smiles or molecule.GetNumAtoms() <= 1:
        raise ValueError("Standardized structure is empty or single-atom")
    try:
        structure_id = Chem.MolToInchiKey(molecule)
    except Exception:
        structure_id = ""
    if not structure_id:
        structure_id = "SMILES_" + sha256_text(model_smiles)[:24]
    return {
        "model_smiles": model_smiles,
        "parent_structure_id": structure_id,
        "parent_selection_method": method,
        "model_atom_count": molecule.GetNumAtoms(),
        "model_carbon_atom_count": sum(
            1 for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 6
        ),
    }


def prepare_bbb_candidate_rows(
    source_csv: Path, output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(source_csv, low_memory=False)
    if "chembl_id" not in frame.columns:
        frame["chembl_id"] = [f"CHEMBL_ROW_{index + 1}" for index in range(len(frame))]
    frame["screening_compound_id"] = np.arange(1, len(frame) + 1, dtype=np.int64)
    parent_column = (
        frame["hierarchy_parent_canonical_smiles"]
        if "hierarchy_parent_canonical_smiles" in frame.columns
        else pd.Series("", index=frame.index)
    )
    prepared: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row, parent_smiles in tqdm(
        zip(frame.to_dict("records"), parent_column.tolist()),
        total=len(frame), desc="BBB structure preparation", unit="molecule",
    ):
        try:
            prepared.append({
                **row,
                **standardize_bbb_structure(row.get("canonical_smiles", ""), parent_smiles),
            })
        except Exception as exc:
            failures.append({
                "screening_compound_id": row.get("screening_compound_id"),
                "chembl_id": row.get("chembl_id", ""),
                "canonical_smiles": row.get("canonical_smiles", ""),
                "error": f"{type(exc).__name__}: {exc}",
            })
    prepared_frame = pd.DataFrame(prepared)
    failure_frame = pd.DataFrame(failures)
    if prepared_frame.empty:
        raise RuntimeError("No ChEMBL structure can be prepared for BP-NET")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(prepared_frame, output_dir / "01_bbb_input_structures.csv")
    atomic_write_csv(failure_frame, output_dir / "01_bbb_input_failures.csv")
    summary = {
        "source_rows": len(frame),
        "prepared_rows": len(prepared_frame),
        "technical_failures": len(failure_frame),
        "unique_model_smiles": int(prepared_frame["model_smiles"].nunique()),
        "scientific_property_filters": [],
    }
    return prepared_frame, failure_frame, summary


def parse_integer_set(value: str, minimum: int, maximum: int, label: str) -> tuple[int, ...]:
    result: list[int] = []
    for token in str(value).replace("\uFF0C", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"Invalid {label} range: {token}")
            result.extend(range(start, end + 1))
        else:
            number = float(token)
            if not number.is_integer():
                raise ValueError(f"{label} must contain integers: {token}")
            result.append(int(number))
    normalized = tuple(sorted(set(result)))
    if not normalized or any(item < minimum or item > maximum for item in normalized):
        raise ValueError(f"{label} must be integers in [{minimum}, {maximum}]")
    return normalized


def parse_seeds(value: str) -> tuple[int, ...]:
    return parse_integer_set(value, 0, 2**31 - 1, "seeds")


def clean_text_series(series: pd.Series) -> pd.Series:
    result = series.fillna("").astype(str).str.strip()
    return result.mask(result.str.casefold().isin({"nan", "none", "null"}), "")


def truthy_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna("").astype(str).str.strip().str.casefold().isin(
        {"true", "1", "yes", "y", "t"}
    )


def external_compound_mask(frame: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for column in (
        "is_external_control",
        "is_known_positive_control",
        "positive_control_bypassed_pipeline_filters",
    ):
        if column in frame.columns:
            mask |= truthy_series(frame[column])
    if "candidate_origin" in frame.columns:
        origin = clean_text_series(frame["candidate_origin"]).str.casefold()
        mask |= origin.str.contains(
            r"bindingdb|external_control|known_positive_control_added_outside_normal_pipeline",
            regex=True,
            na=False,
        )
    if "external_control_id" in frame.columns:
        mask |= clean_text_series(frame["external_control_id"]).ne("")
    if "chembl_id" in frame.columns:
        mask |= clean_text_series(frame["chembl_id"]).str.upper().str.startswith("BDBCTRL_")
    return mask.fillna(False)


def drop_external_metadata_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column
        for column in frame.columns
        if column in EXTERNAL_METADATA_EXACT_COLUMNS
        or column.startswith("external_control_")
        or column.startswith("bindingdb_")
        or column.startswith("positive_control_")
    ]
    return frame.drop(columns=columns, errors="ignore")


def candidate_fingerprint(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(f"rows={len(frame)}|schema=1\n".encode("utf-8"))
    for row in frame.to_dict("records"):
        digest.update(
            ("\x1f".join(
                [
                    str(row.get("chembl_id", "")),
                    str(row.get("parent_structure_id", "")),
                    str(row.get("dta_smiles", "")),
                ]
            ) + "\n").encode("utf-8")
        )
    return digest.hexdigest()


def validate_input_max_phases(
    frame: pd.DataFrame, selected: tuple[int, ...], source: Path
) -> tuple[int, ...]:
    if "max_phase" not in frame.columns:
        raise ValueError(f"Candidate CSV is missing max_phase: {source}")
    numeric = pd.to_numeric(frame["max_phase"], errors="coerce")
    invalid = numeric.isna() | (numeric.sub(numeric.round()).abs() > 1e-8)
    if bool(invalid.any()):
        examples = frame.loc[invalid, ["max_phase"]].head(10).to_dict("records")
        raise ValueError(f"Missing/non-integer max_phase values: {examples}")
    observed = tuple(sorted(set(numeric.astype(int).tolist())))
    unexpected = sorted(set(observed) - set(selected))
    if unexpected:
        raise RuntimeError(
            "Candidate CSV does not match the exact requested max_phase set.\n"
            f"requested={list(selected)}, observed={list(observed)}, "
            f"unexpected={unexpected}, source={source}"
        )
    return observed


def resolve_smiles_column(frame: pd.DataFrame, args: argparse.Namespace) -> str:
    requested = str(args.smiles_column).strip()
    if requested.casefold() == "auto":
        requested = "model_smiles" if "model_smiles" in frame.columns else "canonical_smiles"
    if requested in frame.columns:
        if requested != "model_smiles" and not args.allow_canonical_smiles_fallback:
            raise ValueError(
                f"Resolved SMILES column is {requested!r}, but the formal pipeline "
                "requires model_smiles. Use --allow-canonical-smiles-fallback only "
                "when the alternative column was standardized identically."
            )
        return requested
    if args.allow_canonical_smiles_fallback and "canonical_smiles" in frame.columns:
        print(
            "[WARNING] model_smiles is unavailable; canonical_smiles will be used "
            "as requested."
        )
        return "canonical_smiles"
    raise ValueError(
        f"Candidate CSV is missing required SMILES column {requested!r}. "
        "Use the parent-deduplicated BBB-permeable primary CSV."
    )


def load_candidates(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = args.candidate_csv
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing/empty candidate CSV: {path}")
    frame = pd.read_csv(path, low_memory=False)
    source_rows = len(frame)
    source_hash = sha256_file(path)
    observed_phases = validate_input_max_phases(frame, args.max_phases, path)

    external = external_compound_mask(frame)
    external_removed = int(external.sum())
    frame = drop_external_metadata_columns(frame.loc[~external].copy())
    smiles_column = resolve_smiles_column(frame, args)

    if "screening_compound_id" not in frame.columns:
        frame["screening_compound_id"] = np.arange(1, len(frame) + 1, dtype=np.int64)
    numeric_ids = pd.to_numeric(frame["screening_compound_id"], errors="coerce")
    if numeric_ids.isna().any() or numeric_ids.duplicated().any():
        numeric_ids = pd.Series(np.arange(1, len(frame) + 1), index=frame.index)
    frame["screening_compound_id"] = numeric_ids.astype(np.int64)

    if "chembl_id" not in frame.columns:
        frame["chembl_id"] = frame["screening_compound_id"].map(
            lambda value: f"CHEMBL_ROW_{int(value)}"
        )
    frame["chembl_id"] = clean_text_series(frame["chembl_id"])
    missing_ids = frame["chembl_id"].eq("")
    frame.loc[missing_ids, "chembl_id"] = frame.loc[
        missing_ids, "screening_compound_id"
    ].map(lambda value: f"CHEMBL_ROW_{int(value)}")

    if "pref_name" not in frame.columns:
        frame["pref_name"] = ""
    frame["pref_name"] = clean_text_series(frame["pref_name"])
    if "canonical_smiles" not in frame.columns:
        frame["canonical_smiles"] = ""
    frame["canonical_smiles"] = clean_text_series(frame["canonical_smiles"])
    if "model_smiles" in frame.columns:
        frame["model_smiles"] = clean_text_series(frame["model_smiles"])
    frame["dta_smiles"] = clean_text_series(frame[smiles_column])
    missing_smiles = frame["dta_smiles"].eq("")
    missing_smiles_removed = int(missing_smiles.sum())
    frame = frame.loc[~missing_smiles].copy()

    if "hierarchy_parent_pref_name" in frame.columns:
        parent_name = clean_text_series(frame["hierarchy_parent_pref_name"])
        frame["dta_pref_name"] = parent_name.where(parent_name.ne(""), frame["pref_name"])
    else:
        frame["dta_pref_name"] = frame["pref_name"]

    before_bbb = len(frame)
    if args.bbb_group:
        if "bbb_confidence_group" not in frame.columns:
            raise ValueError("--bbb-group requires bbb_confidence_group")
        keep = frame["bbb_confidence_group"].astype(str).eq(args.bbb_group)
    else:
        required = {"bbb_prob_mean", "bbb_pass_votes"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Numeric BBB filtering is missing columns: {sorted(missing)}")
        keep = (
            pd.to_numeric(frame["bbb_prob_mean"], errors="coerce").ge(args.bbb_min_prob)
            & pd.to_numeric(frame["bbb_pass_votes"], errors="coerce").ge(args.bbb_min_votes)
        )
    frame = frame.loc[keep].copy()
    bbb_removed = before_bbb - len(frame)

    black_box_removed = 0
    if args.exclude_black_box_warning:
        if "black_box_warning" in frame.columns:
            before = len(frame)
            flag = pd.to_numeric(frame["black_box_warning"], errors="coerce").fillna(0)
            frame = frame.loc[flag.astype(int).eq(0)].copy()
            black_box_removed = before - len(frame)
        else:
            print(
                "[WARNING] black_box_warning is unavailable; the optional filter "
                "was skipped."
            )

    withdrawn_removed = 0
    if args.exclude_withdrawn and "withdrawn_flag" in frame.columns:
        before = len(frame)
        frame = frame.loc[~truthy_series(frame["withdrawn_flag"])].copy()
        withdrawn_removed = before - len(frame)

    no_carbon_removed = 0
    if args.exclude_no_carbon:
        no_carbon = pd.Series(False, index=frame.index)
        evaluable = False
        if "model_carbon_atom_count" in frame.columns:
            no_carbon |= pd.to_numeric(
                frame["model_carbon_atom_count"], errors="coerce"
            ).eq(0)
            evaluable = True
        if "bbb_ad_no_carbon" in frame.columns:
            no_carbon |= truthy_series(frame["bbb_ad_no_carbon"])
            evaluable = True
        if not evaluable:
            raise ValueError(
                "--exclude-no-carbon requires model_carbon_atom_count or bbb_ad_no_carbon"
            )
        no_carbon_removed = int(no_carbon.sum())
        frame = frame.loc[~no_carbon].copy()

    multicomponent_removed = 0
    if args.exclude_multicomponent:
        multicomponent = frame["dta_smiles"].str.contains(".", regex=False)
        multicomponent_removed = int(multicomponent.sum())
        frame = frame.loc[~multicomponent].copy()

    before_dedup = len(frame)
    if args.deduplicate_by == "dta_smiles":
        frame = frame.drop_duplicates("dta_smiles", keep="first")
    elif args.deduplicate_by == "parent_structure_id":
        if "parent_structure_id" not in frame.columns:
            raise ValueError("parent_structure_id is required by --deduplicate-by")
        parent = clean_text_series(frame["parent_structure_id"])
        frame["_dedup_key"] = np.where(
            parent.ne(""), "parent:" + parent, "smiles:" + frame["dta_smiles"]
        )
        frame = frame.drop_duplicates("_dedup_key", keep="first").drop(columns="_dedup_key")
    elif args.deduplicate_by == "chembl_id":
        frame = frame.drop_duplicates("chembl_id", keep="first")
    elif args.deduplicate_by != "none":
        raise ValueError(f"Unknown deduplication mode: {args.deduplicate_by}")
    dedup_removed = before_dedup - len(frame)

    sort_columns: list[str] = []
    ascending: list[bool] = []
    if "bbb_prob_mean" in frame.columns:
        sort_columns.append("bbb_prob_mean")
        ascending.append(False)
    if "bbb_pass_votes" in frame.columns:
        sort_columns.append("bbb_pass_votes")
        ascending.append(False)
    sort_columns.append("chembl_id")
    ascending.append(True)
    frame = frame.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)

    max_candidates_removed = 0
    if args.max_candidates > 0 and len(frame) > args.max_candidates:
        max_candidates_removed = len(frame) - args.max_candidates
        frame = frame.head(args.max_candidates).copy()
    if frame.empty:
        raise RuntimeError("No candidates remain after filtering")
    if args.expected_candidates > 0 and len(frame) != args.expected_candidates:
        raise RuntimeError(
            f"Candidate count assertion failed: expected={args.expected_candidates}, "
            f"observed={len(frame)}"
        )

    frame["drug_id"] = frame["dta_smiles"].map(
        lambda value: "chembl_smiles_" + sha256_text(str(value))[:24]
    )
    if frame["drug_id"].duplicated().any():
        raise RuntimeError("Duplicate model SMILES remain after candidate selection")
    fingerprint = candidate_fingerprint(frame)
    summary = {
        "source_csv": str(path),
        "source_csv_sha256": source_hash,
        "source_rows": source_rows,
        "selected_max_phases_requested": list(args.max_phases),
        "selected_max_phases_observed": list(observed_phases),
        "external_compound_rows_removed": external_removed,
        "smiles_source_column": smiles_column,
        "invalid_or_missing_dta_smiles_removed": missing_smiles_removed,
        "before_bbb_filter": before_bbb,
        "bbb_filter_removed": bbb_removed,
        "black_box_removed": black_box_removed,
        "withdrawn_removed": withdrawn_removed,
        "no_carbon_removed": no_carbon_removed,
        "multicomponent_dta_smiles_removed": multicomponent_removed,
        "deduplication_mode": args.deduplicate_by,
        "duplicate_inputs_removed": dedup_removed,
        "max_candidates_removed": max_candidates_removed,
        "selected_candidates_before_feature_validation": len(frame),
        "candidate_fingerprint_sha256": fingerprint,
    }
    return frame, summary


def download_nlrp3_fasta(path: Path) -> None:
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
        read_nlrp3_fasta(temporary, allow_mismatch=False, auto_download=False)
        os.replace(temporary, path)
    except (urllib.error.URLError, TimeoutError, OSError, UnicodeError) as exc:
        raise RuntimeError(
            f"Could not download the NLRP3 FASTA from {NLRP3_FASTA_URL}: {exc}"
        ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"[NLRP3] Saved canonical FASTA: {path}")


def read_nlrp3_fasta(
    path: Path, allow_mismatch: bool, auto_download: bool = True
) -> tuple[str, str]:
    if not path.is_file() or path.stat().st_size <= 0:
        if not auto_download:
            raise FileNotFoundError(f"Missing or empty NLRP3 FASTA: {path}")
        download_nlrp3_fasta(path)
    records: list[tuple[str, str]] = []
    header = ""
    pieces: list[str] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header:
                    records.append((header, "".join(pieces)))
                header, pieces = line[1:].strip(), []
            else:
                if not header:
                    raise ValueError("FASTA sequence appears before its header")
                pieces.append(re.sub(r"\s+", "", line).upper())
    if header:
        records.append((header, "".join(pieces)))
    if len(records) != 1:
        raise ValueError("NLRP3 FASTA must contain exactly one record")
    header, sequence = records[0]
    invalid = sorted(set(sequence) - set("ACDEFGHIKLMNPQRSTVWYBXZOUJ"))
    if not sequence or invalid:
        raise ValueError(f"Invalid NLRP3 sequence letters: {invalid}")
    if not allow_mismatch and (
        NLRP3_ACCESSION not in header or len(sequence) != NLRP3_EXPECTED_LENGTH
    ):
        raise RuntimeError(
            "The target FASTA does not look like canonical human NLRP3 Q96P20-1: "
            f"header={header!r}, length={len(sequence)}. "
            "Use --allow-target-mismatch only for an intentional target variant."
        )
    return header, sequence


def set_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def feature_artifact_path(cache_root: Path, entity: Any, feature: str) -> Path:
    entity_type = "ligand" if feature in {"unimol2", "rdkit"} else "protein"
    suffix = ".npz" if feature == "esm2_contact_graph" else ".pt"
    return cache_root / entity_type / feature / f"{entity.artifact_key}{suffix}"


@dataclass
class FeatureScan:
    feature: str
    entities: list[Any]
    rows: dict[str, dict[str, Any]]
    pending: list[Any]
    directory: Path


def scan_features(
    eg: Any,
    embedding_args: argparse.Namespace,
    cache_root: Path,
    feature: str,
    entities: Sequence[Any],
    force: bool,
) -> FeatureScan:
    rows: dict[str, dict[str, Any]] = {}
    pending: list[Any] = []
    directory = cache_root / ("ligand" if feature in {"unimol2", "rdkit"} else "protein") / feature
    for entity in entities:
        path = feature_artifact_path(cache_root, entity, feature)
        valid, detail, shape = eg.validate_artifact(
            path, entity, feature, embedding_args
        )
        if valid and not force:
            rows[entity.entity_id] = eg.manifest_row(
                embedding_args.embedding_root,
                entity,
                feature,
                path,
                "valid",
                detail,
                shape,
            )
        else:
            pending.append(entity)
            rows[entity.entity_id] = eg.manifest_row(
                embedding_args.embedding_root,
                entity,
                feature,
                path,
                "pending",
                "forced" if force else detail,
            )
    return FeatureScan(feature, list(entities), rows, pending, directory)


def update_feature_row(
    eg: Any,
    embedding_args: argparse.Namespace,
    cache_root: Path,
    scan: FeatureScan,
    entity: Any,
    status: str,
    detail: str = "",
) -> None:
    path = feature_artifact_path(cache_root, entity, scan.feature)
    shape = ""
    if status == "valid":
        valid, detail, shape = eg.validate_artifact(
            path, entity, scan.feature, embedding_args
        )
        if not valid:
            raise RuntimeError(
                f"Post-write validation failed for {entity.entity_id}/{scan.feature}: {detail}"
            )
    scan.rows[entity.entity_id] = eg.manifest_row(
        embedding_args.embedding_root,
        entity,
        scan.feature,
        path,
        status,
        detail[:1000],
        shape,
    )


def write_feature_manifest(eg: Any, scan: FeatureScan) -> None:
    eg.write_manifest(scan.directory, scan.rows)


def generate_feature_artifacts(
    eg: Any,
    embedding_args: argparse.Namespace,
    cache_root: Path,
    scans: Mapping[str, FeatureScan],
) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []

    rdkit_scan = scans.get("rdkit")
    if rdkit_scan is not None:
        for index, entity in enumerate(
            tqdm(rdkit_scan.pending, desc="candidate RDKit graphs", unit="compound"), start=1
        ):
            try:
                eg.atomic_torch_save(
                    feature_artifact_path(cache_root, entity, "rdkit"),
                    eg.generate_rdkit_artifact(entity, embedding_args),
                )
                update_feature_row(eg, embedding_args, cache_root, rdkit_scan, entity, "valid")
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                update_feature_row(
                    eg, embedding_args, cache_root, rdkit_scan, entity, "failed", message
                )
                failures.append({
                    "entity_id": entity.entity_id,
                    "feature": "rdkit",
                    "error": message,
                })
            if index % embedding_args.checkpoint_every == 0:
                write_feature_manifest(eg, rdkit_scan)
        write_feature_manifest(eg, rdkit_scan)

    unimol_scan = scans.get("unimol2")
    if unimol_scan is not None and unimol_scan.pending:
        model = eg.load_unimol_model(embedding_args)
        try:
            need_atomic = any(
                entity.dataset in eg.CLASSIFICATION_DATASETS
                for entity in unimol_scan.pending
            )
            ordered = sorted(unimol_scan.pending, key=lambda item: len(item.text))
            batches = [
                ordered[start : start + embedding_args.unimol_batch_size]
                for start in range(0, len(ordered), embedding_args.unimol_batch_size)
            ]
            completed = 0
            for batch in tqdm(batches, desc="candidate UniMol2", unit="batch"):
                try:
                    artifacts = eg.generate_unimol_batch(
                        model, batch, need_atomic, embedding_args
                    )
                    pairs = list(zip(batch, artifacts))
                except Exception as batch_exc:
                    print(
                        "[WARNING] UniMol2 batch inference failed; processing "
                        f"molecules individually: {type(batch_exc).__name__}: {batch_exc}"
                    )
                    pairs = []
                    for entity in batch:
                        try:
                            artifact = eg.generate_unimol_batch(
                                model, [entity], need_atomic, embedding_args
                            )[0]
                            pairs.append((entity, artifact))
                        except Exception as exc:
                            message = f"{type(exc).__name__}: {exc}"
                            update_feature_row(
                                eg,
                                embedding_args,
                                cache_root,
                                unimol_scan,
                                entity,
                                "failed",
                                message,
                            )
                            failures.append({
                                "entity_id": entity.entity_id,
                                "feature": "unimol2",
                                "error": message,
                            })
                for entity, artifact in pairs:
                    eg.atomic_torch_save(
                        feature_artifact_path(cache_root, entity, "unimol2"), artifact
                    )
                    update_feature_row(
                        eg, embedding_args, cache_root, unimol_scan, entity, "valid"
                    )
                completed += len(batch)
                if completed % embedding_args.checkpoint_every == 0:
                    write_feature_manifest(eg, unimol_scan)
            write_feature_manifest(eg, unimol_scan)
        finally:
            eg.clear_model_memory(model)
    elif unimol_scan is not None:
        write_feature_manifest(eg, unimol_scan)

    esmc_scan = scans.get("esmc")
    if esmc_scan is not None and esmc_scan.pending:
        model = eg.load_esmc_model(embedding_args)
        try:
            for entity in esmc_scan.pending:
                try:
                    artifact = eg.generate_esmc_artifact(model, entity, embedding_args)
                    eg.atomic_torch_save(
                        feature_artifact_path(cache_root, entity, "esmc"), artifact
                    )
                    update_feature_row(
                        eg, embedding_args, cache_root, esmc_scan, entity, "valid"
                    )
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    update_feature_row(
                        eg, embedding_args, cache_root, esmc_scan, entity, "failed", message
                    )
                    failures.append({
                        "entity_id": entity.entity_id,
                        "feature": "esmc",
                        "error": message,
                    })
            write_feature_manifest(eg, esmc_scan)
        finally:
            eg.clear_model_memory(model)
    elif esmc_scan is not None:
        write_feature_manifest(eg, esmc_scan)

    graph_scan = scans.get("esm2_contact_graph")
    if graph_scan is not None and graph_scan.pending:
        predictor = eg.load_esm2_predictor(embedding_args)
        try:
            for entity in graph_scan.pending:
                try:
                    arrays, _ = eg.generate_esm2_graph(
                        entity, predictor, embedding_args
                    )
                    eg.atomic_npz_save(
                        feature_artifact_path(
                            cache_root, entity, "esm2_contact_graph"
                        ),
                        **arrays,
                    )
                    update_feature_row(
                        eg, embedding_args, cache_root, graph_scan, entity, "valid"
                    )
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    update_feature_row(
                        eg, embedding_args, cache_root, graph_scan, entity, "failed", message
                    )
                    failures.append({
                        "entity_id": entity.entity_id,
                        "feature": "esm2_contact_graph",
                        "error": message,
                    })
            write_feature_manifest(eg, graph_scan)
        finally:
            eg.clear_model_memory(predictor)
    elif graph_scan is not None:
        write_feature_manifest(eg, graph_scan)
    return failures


def build_embedding_args(args: argparse.Namespace, eg: Any) -> argparse.Namespace:
    values = eg.build_parser().parse_args([])
    values.embedding_root = args.embedding_root
    values.device = eg.resolve_device(args.feature_device)
    values.offline = args.offline
    values.no_offline_first = args.no_offline_first
    values.hf_endpoint = args.hf_endpoint
    values.checkpoint_every = args.feature_checkpoint_every
    values.unimol_model_name = args.unimol_model_name
    values.unimol_model_size = args.unimol_model_size
    values.unimol_batch_size = args.unimol_batch_size
    values.esmc_model_name = args.esmc_model_name
    values.esmc_max_sequence_length = args.esmc_max_sequence_length
    values.esm2_backend = args.esm2_backend
    values.esm2_model_name = args.esm2_model_name
    values.esm2_fair_model_name = args.esm2_fair_model_name
    values.esm2_torch_dtype = args.esm2_torch_dtype
    values.fair_esm_repo_path = args.fair_esm_repo_path
    values.esm2_window_size = args.esm2_window_size
    values.esm2_window_overlap = args.esm2_window_overlap
    values.esm2_probability_threshold = args.esm2_probability_threshold
    values.esm2_top_k_long_range = args.esm2_top_k_long_range
    values.esm2_minimum_separation = args.esm2_minimum_separation
    return values


def manifest_for(cache_root: Path, entity_type: str, feature: str) -> Path:
    return cache_root / entity_type / feature / "manifest.csv"


def verify_training_feature_configurations(
    eg: Any,
    embedding_args: argparse.Namespace,
    dataset: str = "BindingDB",
    features: Sequence[str] = ("unimol2", "rdkit", "esmc", "esm2_contact_graph"),
) -> dict[str, Any]:
    """Require screening features to match those used for model training."""
    result: dict[str, Any] = {}
    for feature in features:
        entity_type = "ligand" if feature in {"unimol2", "rdkit"} else "protein"
        path = (
            embedding_args.embedding_root
            / dataset
            / entity_type
            / feature
            / "manifest.csv"
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {dataset} training feature manifest: {path}. "
                "The screening script cannot verify model-input compatibility."
            )
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        required = {"feature", "status", "configuration_sha256"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        valid = frame.loc[
            frame["feature"].eq(feature) & frame["status"].eq("valid")
        ]
        observed = sorted(
            value for value in valid["configuration_sha256"].unique() if value
        )
        expected = eg.configuration_sha256(
            embedding_args, dataset, feature
        )
        if observed != [expected]:
            raise RuntimeError(
                f"Screening {feature} configuration does not match the {dataset} "
                f"training artifacts: expected={expected}, observed={observed}, "
                f"manifest={path}"
            )
        result[feature] = {
            "training_manifest": str(path),
            "configuration_sha256": expected,
            "valid_training_artifacts": int(len(valid)),
        }
    return result


def find_checkpoint(args: argparse.Namespace, seed: int) -> Path:
    directory = (
        args.model_root
        / "BindingDB"
        / args.run_name
        / f"seed_{seed:04d}"
        / args.scenario
    )
    for name in ("best.pt", "best.pth"):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Missing UE-AlignNet checkpoint for seed={seed}, scenario={args.scenario}: "
        f"{directory / 'best.pt'}"
    )


def inspect_checkpoints(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    checkpoints: list[dict[str, Any]] = []
    reference_architecture: dict[str, Any] | None = None
    reference_effective: dict[str, Any] | None = None
    for seed in args.seeds:
        path = find_checkpoint(args, seed)
        checkpoint = torch_load(path)
        config = checkpoint.get("config", {})
        if config.get("dataset") != "BindingDB":
            raise RuntimeError(f"Checkpoint is not a BindingDB model: {path}")
        if config.get("scenario") != args.scenario:
            raise RuntimeError(f"Checkpoint scenario mismatch: {path}")
        architecture = dict(config.get("architecture", {}))
        effective = dict(config.get("effective", {}))
        state = checkpoint.get("state", {})
        if not architecture or "train_mean" not in state or "train_std" not in state:
            raise RuntimeError(f"Incomplete UE-AlignNet checkpoint metadata: {path}")
        if reference_architecture is None:
            reference_architecture = architecture
            reference_effective = effective
        elif architecture != reference_architecture or effective != reference_effective:
            raise RuntimeError(
                "Ensemble checkpoints use different architectures/effective feature settings"
            )
        checkpoints.append({
            "seed": seed,
            "path": path,
            "sha256": sha256_file(path),
            "train_mean": float(state["train_mean"]),
            "train_std": float(state["train_std"]),
        })
        del checkpoint
    assert reference_architecture is not None and reference_effective is not None
    return checkpoints, reference_architecture, reference_effective


def build_stores(
    ue: Any,
    args: argparse.Namespace,
    cache_root: Path,
    architecture: Mapping[str, Any],
    effective: Mapping[str, Any],
    ligand_ids: Sequence[str],
) -> tuple[Any, Any, Any]:
    ligand_index = ue.ArtifactIndex(
        args.embedding_root,
        manifest_for(cache_root, "ligand", "unimol2"),
        "unimol2",
    )
    rdkit_index = ue.ArtifactIndex(
        args.embedding_root,
        manifest_for(cache_root, "ligand", "rdkit"),
        "rdkit",
    )
    protein_index = ue.ArtifactIndex(
        args.embedding_root,
        manifest_for(cache_root, "protein", "esmc"),
        "esmc",
    )
    graph_index = ue.ArtifactIndex(
        args.embedding_root,
        manifest_for(cache_root, "protein", "esm2_contact_graph"),
        "esm2_contact_graph",
    )
    ligand_index.require(ligand_ids)
    rdkit_index.require(ligand_ids)
    protein_index.require([NLRP3_ENTITY_ID])
    graph_index.require([NLRP3_ENTITY_ID])
    max_ligand_atoms = int(effective.get("max_ligand_atoms") or 0) or None
    return (
        ue.LigandStore(
            ligand_index,
            rdkit_index,
            int(architecture["ligand_dim"]),
            int(architecture["graph_in_dim"]),
            args.ligand_cache_size,
            max_ligand_atoms,
        ),
        ue.ProteinStore(
            protein_index,
            int(architecture["protein_dim"]),
            1,
        ),
        ue.ProteinGraphStore(
            graph_index,
            str(effective.get("protein_graph_mode", "contact_plus_band")),
            int(effective.get("protein_sequence_band_width", 2)),
            float(effective.get("protein_contact_threshold", 0.0)),
            1,
        ),
    )


def prediction_part_valid(
    path: Path,
    expected_ids: Sequence[str],
    run_fingerprint: str,
) -> bool:
    try:
        frame = pd.read_csv(path, dtype={"drug_id": str})
        return bool(
            list(frame.columns) == ["run_fingerprint", "drug_id", "pred_pKd"]
            and frame["run_fingerprint"].astype(str).eq(run_fingerprint).all()
            and frame["drug_id"].astype(str).tolist() == list(expected_ids)
            and np.isfinite(pd.to_numeric(frame["pred_pKd"], errors="coerce")).all()
        )
    except Exception:
        return False


@torch.no_grad()
def predict_checkpoint_chunks(
    ue: Any,
    args: argparse.Namespace,
    checkpoint_info: Mapping[str, Any],
    architecture: Mapping[str, Any],
    effective: Mapping[str, Any],
    candidates: pd.DataFrame,
    stores: tuple[Any, Any, Any],
    parts_dir: Path,
    run_fingerprint: str,
    device: torch.device,
) -> None:
    checkpoint = torch_load(Path(checkpoint_info["path"]), map_location=device)
    model = ue.UEAlignNet12A(**dict(architecture)).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    del checkpoint
    max_protein_len = effective.get("max_protein_len")
    if max_protein_len is not None:
        max_protein_len = int(max_protein_len)
        if max_protein_len == 0:
            max_protein_len = None
    seed = int(checkpoint_info["seed"])

    chunks = [
        candidates.iloc[start : start + args.chunk_size].copy().reset_index(drop=True)
        for start in range(0, len(candidates), args.chunk_size)
    ]
    for chunk_index, chunk in enumerate(
        tqdm(chunks, desc=f"UE-AlignNet seed {seed}", unit="chunk"), start=1
    ):
        part_path = parts_dir / f"seed_{seed:04d}_chunk_{chunk_index:06d}.csv"
        expected_ids = chunk["drug_id"].astype(str).tolist()
        if not args.overwrite and prediction_part_valid(
            part_path, expected_ids, run_fingerprint
        ):
            continue
        inference_frame = pd.DataFrame({
            "drug_id": expected_ids,
            "target_id": NLRP3_ENTITY_ID,
            "affinity": np.zeros(len(chunk), dtype=np.float32),
        })
        dataset = ue.InteractionDataset(
            inference_frame,
            *stores,
            0.0,
            1.0,
            max_protein_len,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            collate_fn=ue.PadCollate(),
        )
        predictions: list[np.ndarray] = []
        observed_ids: list[str] = []
        for batch in loader:
            observed_ids.extend(batch["drug_ids"])
            batch = ue.move_batch_to_device(batch, device)
            with ue.autocast_context(device, args.amp):
                normalized = model(
                    batch["ligand"],
                    batch["graph_x"],
                    batch["graph_adj"],
                    batch["graph_mask"],
                    batch["protein"],
                    batch["protein_adj"],
                    batch["mask"],
                )
            pred_pkd = (
                normalized.float().cpu().numpy() * float(checkpoint_info["train_std"])
                + float(checkpoint_info["train_mean"])
            )
            predictions.append(pred_pkd.astype(np.float32))
        if observed_ids != expected_ids:
            raise RuntimeError("Prediction DataLoader changed candidate ordering")
        values = np.concatenate(predictions) if predictions else np.empty(0, np.float32)
        if len(values) != len(expected_ids) or not np.isfinite(values).all():
            raise RuntimeError(f"Invalid predictions for seed={seed}, chunk={chunk_index}")
        atomic_write_csv(
            pd.DataFrame({
                "run_fingerprint": run_fingerprint,
                "drug_id": expected_ids,
                "pred_pKd": values,
            }),
            part_path,
        )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def collect_seed_predictions(
    candidates: pd.DataFrame,
    parts_dir: Path,
    seeds: Sequence[int],
    chunk_size: int,
    run_fingerprint: str,
) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    expected_all = candidates["drug_id"].astype(str).tolist()
    chunk_count = math.ceil(len(candidates) / chunk_size)
    for seed in seeds:
        ids: list[str] = []
        values: list[np.ndarray] = []
        for chunk_index in range(1, chunk_count + 1):
            path = parts_dir / f"seed_{seed:04d}_chunk_{chunk_index:06d}.csv"
            frame = pd.read_csv(path, dtype={"drug_id": str})
            expected_chunk = expected_all[
                (chunk_index - 1) * chunk_size : chunk_index * chunk_size
            ]
            if not prediction_part_valid(path, expected_chunk, run_fingerprint):
                raise RuntimeError(f"Invalid/incomplete prediction part: {path}")
            ids.extend(frame["drug_id"].astype(str).tolist())
            values.append(frame["pred_pKd"].to_numpy(np.float32))
        if ids != expected_all:
            raise RuntimeError(f"Prediction parts are misaligned for seed={seed}")
        result[seed] = np.concatenate(values)
    return result


def build_results(
    candidates: pd.DataFrame,
    predictions: Mapping[int, np.ndarray],
    args: argparse.Namespace,
    sequence: str,
) -> pd.DataFrame:
    frame = candidates.drop(columns=["drug_id"], errors="ignore").copy()
    frame.insert(0, "target_name", "NLRP3")
    frame.insert(1, "target_accession", NLRP3_ENTITY_ID)
    frame.insert(2, "target_sequence_sha256", sha256_text(sequence))
    frame.insert(3, "target_sequence_length", len(sequence))
    frame.insert(4, "training_dataset", "BindingDB")
    frame.insert(5, "scenario", args.scenario)
    prediction_columns: list[str] = []
    for seed in sorted(predictions):
        column = f"pred_pKd_seed{seed}"
        frame[column] = predictions[seed].astype(np.float32)
        prediction_columns.append(column)
    matrix = frame[prediction_columns].to_numpy(np.float32)
    frame["pred_pKd_mean"] = matrix.mean(axis=1)
    frame["pred_pKd_std"] = matrix.std(axis=1, ddof=0)
    frame["pred_pKd_cv"] = np.where(
        np.abs(frame["pred_pKd_mean"]) > 1e-8,
        frame["pred_pKd_std"] / frame["pred_pKd_mean"],
        np.nan,
    )
    frame["pred_pKd_min"] = matrix.min(axis=1)
    frame["pred_pKd_max"] = matrix.max(axis=1)
    exponent = np.clip(9.0 - frame["pred_pKd_mean"].to_numpy(float), -300, 300)
    frame["pred_Kd_nM_from_mean_pKd"] = np.power(10.0, exponent)
    frame["strong_bind_votes"] = (
        matrix >= args.strong_pkd_threshold
    ).sum(axis=1).astype(np.int16)
    frame["strong_bind_fraction"] = frame["strong_bind_votes"] / len(prediction_columns)
    strong_votes = max(1, len(prediction_columns) - 1)
    frame["binding_confidence_group"] = np.select(
        [
            frame["pred_pKd_mean"].ge(args.strong_pkd_threshold)
            & frame["strong_bind_votes"].ge(strong_votes),
            frame["pred_pKd_mean"].lt(args.strong_pkd_threshold)
            & frame["strong_bind_votes"].le(1),
        ],
        ["high_confidence_strong_binder", "likely_weak_or_non_binder"],
        default="uncertain_or_borderline",
    )
    frame = frame.sort_values(
        ["pred_pKd_mean", "pred_pKd_std"], ascending=[False, True]
    ).reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1, dtype=np.int64))
    return frame


def clear_prediction_outputs(output_dir: Path) -> None:
    for name in (
        "all_predictions.csv",
        "top_candidates.csv",
        "high_confidence_strong_binders.csv",
        "run_manifest.json",
    ):
        path = output_dir / name
        if path.is_file():
            path.unlink()
    parts = output_dir / "parts"
    if parts.is_dir():
        for path in parts.glob("seed_*_chunk_*.csv"):
            path.unlink()


def write_audit_outputs(
    output_dir: Path,
    candidates: pd.DataFrame,
    failure_frame: pd.DataFrame,
    selection_summary: Mapping[str, Any],
) -> None:
    atomic_write_csv(
        candidates.drop(columns="drug_id", errors="ignore"),
        output_dir / "selected_candidates.csv",
    )
    atomic_write_csv(failure_frame, output_dir / "feature_failures.csv")
    atomic_write_json(
        output_dir / "candidate_selection_summary.json", selection_summary
    )


def binary_threshold_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int8)
    p = np.asarray(probabilities, dtype=np.float64)
    predicted = p >= float(threshold)
    tp = int(((y == 1) & predicted).sum())
    fp = int(((y == 0) & predicted).sum())
    tn = int(((y == 0) & ~predicted).sum())
    fn = int(((y == 1) & ~predicted).sum())

    def ratio(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else float("nan")

    return {
        "threshold": float(threshold), "n": len(y),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "ppv": ratio(tp, tp + fp), "npv": ratio(tn, tn + fn),
        "sensitivity": ratio(tp, tp + fn), "specificity": ratio(tn, tn + fp),
        "positive_selected": tp + fp, "negative_selected": tn + fn,
    }


def build_bbb_threshold_curve(
    labels: np.ndarray, probabilities: np.ndarray, pivot: float, grid_size: int
) -> pd.DataFrame:
    y = np.asarray(labels, dtype=np.int8)
    p = np.asarray(probabilities, dtype=np.float64)
    if len(y) == 0 or len(y) != len(p) or len(np.unique(y)) != 2:
        raise ValueError("BBB threshold calibration requires both classes")
    if not np.isfinite(p).all():
        raise ValueError("BBB calibration probabilities contain NaN/Inf")

    def values(low: float, high: float) -> np.ndarray:
        candidates = np.unique(p[(p >= low) & (p <= high)])
        if len(candidates) > grid_size:
            candidates = np.unique(
                np.quantile(candidates, np.linspace(0.0, 1.0, grid_size))
            )
        return np.unique(np.concatenate([candidates, [low, high]]))

    rows: list[dict[str, Any]] = []
    for direction, candidates in (
        ("positive", values(pivot, 1.0)),
        ("negative", values(0.0, pivot)),
    ):
        for threshold in candidates:
            applied = (
                float(threshold)
                if direction == "positive"
                else float(np.nextafter(threshold, np.inf))
            )
            metrics = binary_threshold_metrics(y, p, applied)
            metrics["threshold"] = float(threshold)
            if direction == "positive":
                selected = metrics["positive_selected"]
                quality = metrics["ppv"]
                coverage = metrics["sensitivity"]
            else:
                selected = metrics["negative_selected"]
                quality = metrics["npv"]
                coverage = metrics["specificity"]
            rows.append({
                "direction": direction, **metrics,
                "n_selected": selected,
                "selected_fraction": selected / len(y),
                "quality_value": quality,
                "class_coverage": coverage,
            })
    return pd.DataFrame(rows)


def choose_bbb_threshold(
    curve: pd.DataFrame, direction: str, target: float, min_samples: int
) -> dict[str, Any]:
    part = curve.loc[
        curve["direction"].eq(direction) & np.isfinite(curve["quality_value"])
    ].copy()
    candidates = part.loc[part["n_selected"].ge(min_samples)].copy()
    size_met = not candidates.empty
    if candidates.empty:
        candidates = part.loc[part["n_selected"].gt(0)].copy()
    if candidates.empty:
        raise RuntimeError(f"No usable {direction} BBB threshold")
    target_rows = candidates.loc[candidates["quality_value"].ge(target)].copy()
    if target_rows.empty:
        selected = candidates.sort_values(
            ["quality_value", "n_selected", "class_coverage"],
            ascending=[False, False, False],
        ).iloc[0]
        target_met = False
        reason = "target_not_met_best_available_quality"
    else:
        selected = target_rows.sort_values(
            ["n_selected", "quality_value", "class_coverage"],
            ascending=[False, False, False],
        ).iloc[0]
        target_met = True
        reason = "target_met_maximum_selected_coverage"
    result = {
        key: json_safe(selected[key])
        for key in (
            "threshold", "n", "n_selected", "selected_fraction", "quality_value",
            "class_coverage", "tp", "fp", "tn", "fn", "ppv", "npv",
            "sensitivity", "specificity",
        )
    }
    result.update({
        "direction": direction,
        "target": target,
        "target_met": target_met,
        "minimum_selected_samples_requested": min_samples,
        "minimum_selected_samples_met": size_met,
        "selection_reason": reason,
    })
    return result


def locate_bp_checkpoint(args: argparse.Namespace, seed: int) -> Path:
    root = args.bp_model_root / "B3DB" / args.bp_run_name
    candidates = (
        root / f"seed_{seed:04d}" / "best_model.pt",
        root / f"seed_{seed}" / "best_model.pt",
    )
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(
        f"Missing B3DB BP-NET checkpoint for seed={seed}; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def bp_input_fingerprint(frame: pd.DataFrame) -> str:
    logical = frame[["SMILES", "label"]].copy()
    logical["SMILES"] = logical["SMILES"].astype(str)
    return sha256_text(logical.to_csv(index=False))


def load_cached_bp_probabilities(
    path: Path, frame: pd.DataFrame, fingerprint: str
) -> np.ndarray | None:
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    try:
        cached = pd.read_csv(path, dtype={"SMILES": str, "run_fingerprint": str})
        required = {"row_index", "SMILES", "probability", "run_fingerprint"}
        if not required.issubset(cached) or len(cached) != len(frame):
            return None
        if not bool(cached["run_fingerprint"].eq(fingerprint).all()):
            return None
        if cached["row_index"].tolist() != list(range(len(frame))):
            return None
        if cached["SMILES"].astype(str).tolist() != frame["SMILES"].astype(str).tolist():
            return None
        values = pd.to_numeric(cached["probability"], errors="coerce").to_numpy(float)
        return values.astype(np.float32) if np.isfinite(values).all() else None
    except Exception:
        return None


def predict_bp_frame(
    bp: Any,
    model: torch.nn.Module,
    frame: pd.DataFrame,
    store: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    dataset = bp.BBBUnifiedDataset(frame, store)
    loader_options: dict[str, Any] = {
        "batch_size": args.bp_batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": bp.collate_unified,
    }
    if args.num_workers > 0:
        loader_options["persistent_workers"] = True
    output = bp.run_eval(model, DataLoader(dataset, **loader_options), device)
    if output.smiles != frame["SMILES"].astype(str).tolist():
        raise RuntimeError("BP-NET DataLoader changed molecule ordering")
    return output.probs.astype(np.float32)


def run_bp_net_screening(
    args: argparse.Namespace,
    bp: Any,
    eg: Any,
    embedding_args: argparse.Namespace,
    device: torch.device,
    small_molecule_csv: Path,
) -> tuple[Path, dict[str, Any]]:
    prepared, structure_failures, input_summary = prepare_bbb_candidate_rows(
        small_molecule_csv, args.bbb_output_dir
    )
    bp_training_feature_configurations = verify_training_feature_configurations(
        eg, embedding_args, "B3DB", ("unimol2", "rdkit")
    )
    unique_smiles = prepared["model_smiles"].drop_duplicates().astype(str).tolist()
    entities = [
        eg.Entity("B3DB", "ligand", bp.entity_id_for_smiles(smiles), smiles)
        for smiles in unique_smiles
    ]
    cache_root = args.bp_feature_cache_root.resolve()
    scans = {
        feature: scan_features(
            eg, embedding_args, cache_root, feature, entities, args.force_features
        )
        for feature in ("unimol2", "rdkit")
    }
    pending = {feature: len(scan.pending) for feature, scan in scans.items()}
    if args.check_only and any(pending.values()):
        raise FileNotFoundError(f"Missing BP-NET candidate features: {pending}")
    feature_failures = [] if args.check_only else generate_feature_artifacts(
        eg, embedding_args, cache_root, scans
    )
    atomic_write_csv(
        pd.DataFrame(feature_failures, columns=["entity_id", "feature", "error"]),
        args.bbb_output_dir / "01_bbb_feature_failures.csv",
    )
    failed_ids = {
        entity_id
        for scan in scans.values()
        for entity_id, row in scan.rows.items()
        if row["status"] != "valid"
    }
    prepared["bp_entity_id"] = prepared["model_smiles"].map(bp.entity_id_for_smiles)
    eligible = prepared.loc[~prepared["bp_entity_id"].isin(failed_ids)].copy()
    if eligible.empty:
        raise RuntimeError("All ChEMBL molecules failed BP-NET feature generation")

    training_unimol = bp.ArtifactIndex(
        args.embedding_root,
        bp.manifest_path(args.embedding_root, "B3DB", "unimol2"),
        "unimol2",
    )
    training_rdkit = bp.ArtifactIndex(
        args.embedding_root,
        bp.manifest_path(args.embedding_root, "B3DB", "rdkit"),
        "rdkit",
    )
    screening_unimol = bp.ArtifactIndex(
        args.embedding_root, manifest_for(cache_root, "ligand", "unimol2"), "unimol2"
    )
    screening_rdkit = bp.ArtifactIndex(
        args.embedding_root, manifest_for(cache_root, "ligand", "rdkit"), "rdkit"
    )
    screening_ids = sorted(set(eligible["bp_entity_id"].astype(str)))
    screening_unimol.require(screening_ids)
    screening_rdkit.require(screening_ids)
    training_store = bp.FeatureStore(
        training_unimol, training_rdkit, args.bp_feature_cache_size
    )
    screening_store = bp.FeatureStore(
        screening_unimol, screening_rdkit, args.bp_feature_cache_size
    )
    # Touch one artifact from each source before loading any checkpoint.
    shared_training_ids = sorted(set(training_unimol.paths) & set(training_rdkit.paths))
    if not shared_training_ids:
        raise RuntimeError("B3DB UniMol2 and RDKit manifests have no shared valid molecule")
    training_store.get(shared_training_ids[0])
    screening_store.get(screening_ids[0])

    candidate_frame = eligible[["model_smiles"]].rename(
        columns={"model_smiles": "SMILES"}
    )
    candidate_frame["label"] = 0
    parts_dir = args.bbb_output_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    validation_records: list[pd.DataFrame] = []
    candidate_probabilities: dict[int, np.ndarray] = {}
    checkpoint_report: dict[str, Any] = {}

    for seed in args.bp_seeds:
        checkpoint_path = locate_bp_checkpoint(args, seed)
        checkpoint_hash = sha256_file(checkpoint_path)
        checkpoint = torch_load(checkpoint_path)
        if checkpoint.get("architecture_version") != bp.ARCHITECTURE_VERSION:
            raise RuntimeError(
                f"BP-NET architecture mismatch in {checkpoint_path}: "
                f"{checkpoint.get('architecture_version')!r} != {bp.ARCHITECTURE_VERSION!r}"
            )
        variant = checkpoint.get("variant_config") or bp.VARIANT_CONFIG
        model = bp.BPUnifiedThreeBranch(
            int(checkpoint.get("unimol_atom_dim", bp.UNIMOL_ATOM_DIM)),
            int(checkpoint.get("atom_dim", bp.ATOM_FEAT_DIM)),
            int(checkpoint.get("edge_dim", bp.BOND_FEAT_DIM)),
            variant,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

        split = bp.load_split_bundle(bp.split_paths(args.data_root, "B3DB", seed))["val"]
        validation_fp = sha256_text(
            checkpoint_hash + "|validation|" + bp_input_fingerprint(split)
        )
        validation_path = parts_dir / f"b3db_validation_seed_{seed:04d}.csv"
        val_probs = None if args.force_bbb else load_cached_bp_probabilities(
            validation_path, split, validation_fp
        )
        if val_probs is None:
            training_unimol.require(split["SMILES"].map(bp.entity_id_for_smiles))
            training_rdkit.require(split["SMILES"].map(bp.entity_id_for_smiles))
            val_probs = predict_bp_frame(
                bp, model, split, training_store, args, device
            )
            atomic_write_csv(pd.DataFrame({
                "run_fingerprint": validation_fp,
                "row_index": np.arange(len(split)),
                "SMILES": split["SMILES"],
                "label": split["label"],
                "probability": val_probs,
            }), validation_path)

        candidate_fp = sha256_text(
            checkpoint_hash + "|candidate|" + bp_input_fingerprint(candidate_frame)
        )
        candidate_path = parts_dir / f"chembl_seed_{seed:04d}.csv"
        chembl_probs = None if args.force_bbb else load_cached_bp_probabilities(
            candidate_path, candidate_frame, candidate_fp
        )
        if chembl_probs is None:
            chembl_probs = predict_bp_frame(
                bp, model, candidate_frame, screening_store, args, device
            )
            atomic_write_csv(pd.DataFrame({
                "run_fingerprint": candidate_fp,
                "row_index": np.arange(len(candidate_frame)),
                "SMILES": candidate_frame["SMILES"],
                "label": candidate_frame["label"],
                "probability": chembl_probs,
            }), candidate_path)
        candidate_probabilities[seed] = chembl_probs
        validation_records.append(pd.DataFrame({
            "seed": seed,
            "SMILES": split["SMILES"].astype(str),
            "label": split["label"].astype(int),
            "bbb_probability": val_probs,
        }))
        checkpoint_report[str(seed)] = {
            "path": str(checkpoint_path),
            "sha256": checkpoint_hash,
            "selected_phase": checkpoint.get("selected_phase"),
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    validation_raw = pd.concat(validation_records, ignore_index=True)
    label_counts = validation_raw.groupby("SMILES")["label"].nunique()
    if bool(label_counts.gt(1).any()):
        raise RuntimeError("B3DB validation splits contain conflicting labels for a SMILES")
    validation_oof = validation_raw.groupby(["SMILES", "label"], as_index=False).agg(
        bbb_prob_oof_mean=("bbb_probability", "mean"),
        bbb_prob_oof_std=("bbb_probability", lambda values: float(np.std(values, ddof=0))),
        n_oof_seed_predictions=("seed", "nunique"),
    )
    curve = build_bbb_threshold_curve(
        validation_oof["label"].to_numpy(np.int8),
        validation_oof["bbb_prob_oof_mean"].to_numpy(float),
        args.bbb_vote_threshold,
        args.bbb_threshold_grid_size,
    )
    positive = choose_bbb_threshold(
        curve, "positive", args.bbb_positive_ppv_target, args.bbb_min_threshold_samples
    )
    negative = choose_bbb_threshold(
        curve, "negative", args.bbb_negative_npv_target, args.bbb_min_threshold_samples
    )
    positive_threshold = float(positive["threshold"])
    negative_threshold = float(negative["threshold"])
    min_votes = args.bbb_min_positive_votes
    if min_votes is None:
        min_votes = math.ceil(0.8 * len(args.bp_seeds))
    max_negative_votes = math.floor(0.2 * len(args.bp_seeds))

    output = eligible.drop(columns=["bp_entity_id"], errors="ignore").reset_index(drop=True)
    probability_columns: list[str] = []
    for seed in args.bp_seeds:
        column = f"bbb_prob_seed{seed}"
        output[column] = candidate_probabilities[seed]
        probability_columns.append(column)
    matrix = output[probability_columns].to_numpy(np.float32)
    output["bbb_prob_mean"] = matrix.mean(axis=1)
    output["bbb_prob_std"] = matrix.std(axis=1, ddof=0)
    output["bbb_pass_votes"] = (matrix >= args.bbb_vote_threshold).sum(axis=1)
    output["bbb_pass_fraction"] = output["bbb_pass_votes"] / len(probability_columns)
    output["bbb_class_by_mean"] = output["bbb_prob_mean"].ge(
        args.bbb_vote_threshold
    ).astype(np.int8)
    output["bbb_positive_threshold_from_validation"] = positive_threshold
    output["bbb_negative_threshold_from_validation"] = negative_threshold
    output["bbb_positive_ppv_target_met"] = bool(positive["target_met"])
    output["bbb_negative_npv_target_met"] = bool(negative["target_met"])
    output["bbb_confidence_group"] = np.select(
        [
            output["bbb_prob_mean"].ge(positive_threshold)
            & output["bbb_pass_votes"].ge(min_votes),
            output["bbb_prob_mean"].le(negative_threshold)
            & output["bbb_pass_votes"].le(max_negative_votes),
        ],
        ["high_confidence_BBB_permeable", "high_confidence_BBB_nonpermeable"],
        default="uncertain_or_borderline",
    )
    output["bbb_screen_keep"] = output["bbb_confidence_group"].eq(
        "high_confidence_BBB_permeable"
    )
    output = output.sort_values(
        ["bbb_prob_mean", "bbb_pass_votes"], ascending=[False, False]
    ).reset_index(drop=True)
    output.insert(0, "rank_bbb", np.arange(1, len(output) + 1))
    output["parent_group_size"] = output.groupby("parent_structure_id")[
        "parent_structure_id"
    ].transform("size")
    parent_predictions = output.drop_duplicates("parent_structure_id", keep="first").copy()
    parent_predictions.insert(0, "rank_parent_bbb", np.arange(1, len(parent_predictions) + 1))
    retained = parent_predictions.loc[parent_predictions["bbb_screen_keep"]].copy()
    retained["rank_parent_bbb"] = np.arange(1, len(retained) + 1)

    candidate_path = args.bbb_output_dir / DEFAULT_BBB_CANDIDATE_CSV.name
    atomic_write_csv(validation_raw, args.bbb_output_dir / "02_b3db_validation_predictions.csv")
    atomic_write_csv(validation_oof, args.bbb_output_dir / "02_b3db_validation_oof.csv")
    atomic_write_csv(curve, args.bbb_output_dir / "02_bbb_threshold_curve.csv")
    atomic_write_csv(output, args.bbb_output_dir / "02_bbb_predictions.csv")
    atomic_write_csv(parent_predictions, args.bbb_output_dir / "03_bbb_parent_predictions.csv")
    atomic_write_csv(retained, candidate_path)
    threshold_report = {
        "calibration_dataset": "B3DB validation splits",
        "calibration_unit": "unique SMILES with mean out-of-fold seed probability",
        "positive": positive,
        "negative": negative,
        "vote_threshold": args.bbb_vote_threshold,
        "minimum_positive_votes": min_votes,
        "selected_seeds": list(args.bp_seeds),
    }
    atomic_write_json(args.bbb_output_dir / "02_bbb_thresholds.json", threshold_report)
    manifest = {
        "completed": True,
        "input": input_summary,
        "structure_failures": len(structure_failures),
        "feature_failures": len(feature_failures),
        "training_feature_configuration_checks": bp_training_feature_configurations,
        "checkpoints": checkpoint_report,
        "thresholds": threshold_report,
        "prediction_rows": len(output),
        "parent_prediction_rows": len(parent_predictions),
        "high_confidence_parent_BBB_permeable": len(retained),
        "candidate_csv": str(candidate_path),
        "model_architecture": bp.ARCHITECTURE_VERSION,
    }
    atomic_write_json(args.bbb_output_dir / "bbb_screening_manifest.json", manifest)
    print(
        f"[BP-NET] predictions={len(output):,}, parent BBB+={len(retained):,}, "
        f"positive threshold={positive_threshold:.6f}, votes>={min_votes}"
    )
    if not positive["target_met"]:
        print(
            "[WARNING] The requested BBB-positive PPV was not reached; the best "
            "validation threshold was used."
        )
    if not negative["target_met"]:
        print(
            "[WARNING] The requested BBB-negative NPV was not reached; the best "
            "validation threshold was used."
        )
    if retained.empty:
        raise RuntimeError("BP-NET retained no high-confidence BBB-permeable parent candidates")
    return candidate_path, manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download and prepare ChEMBL max_phase=4 small molecules, screen BBB "
            "permeability with BP-NET, then rank NLRP3 affinity with UE-AlignNet."
        )
    )
    parser.add_argument(
        "--candidate-csv", type=Path, default=None,
        help="Start from an existing BBB prediction CSV and skip ChEMBL/BP-NET stages",
    )
    parser.add_argument("--chembl-root", type=Path, default=DEFAULT_CHEMBL_ROOT)
    parser.add_argument("--chembl-source-csv", type=Path, default=None)
    parser.add_argument(
        "--chembl-small-molecule-csv", type=Path,
        default=DEFAULT_CHEMBL_SMALL_MOLECULE_CSV,
    )
    parser.add_argument("--chembl-page-size", type=int, default=1000)
    parser.add_argument("--download-retries", type=int, default=5)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)

    parser.add_argument("--bbb-output-dir", type=Path, default=DEFAULT_BBB_OUTPUT_DIR)
    parser.add_argument("--bp-model-root", type=Path, default=DEFAULT_BP_MODEL_ROOT)
    parser.add_argument("--bp-run-name", default="default")
    parser.add_argument("--bp-seeds", default="1-5")
    parser.add_argument("--bp-batch-size", type=int, default=64)
    parser.add_argument("--bp-feature-cache-size", type=int, default=4096)
    parser.add_argument(
        "--bp-feature-cache-root", type=Path, default=DEFAULT_BP_FEATURE_CACHE_ROOT
    )
    parser.add_argument("--bbb-vote-threshold", type=float, default=0.5)
    parser.add_argument("--bbb-positive-ppv-target", type=float, default=0.90)
    parser.add_argument("--bbb-negative-npv-target", type=float, default=0.90)
    parser.add_argument("--bbb-min-threshold-samples", type=int, default=20)
    parser.add_argument("--bbb-threshold-grid-size", type=int, default=2001)
    parser.add_argument("--bbb-min-positive-votes", type=int, default=None)
    parser.add_argument("--force-bbb", action="store_true")
    parser.add_argument("--bbb-only", action="store_true")

    parser.add_argument("--nlrp3-fasta", type=Path, default=DEFAULT_NLRP3_FASTA)
    parser.add_argument("--allow-target-mismatch", action="store_true")
    parser.add_argument("--max-phases", default="4", help="Exact ChEMBL max_phase set")
    parser.add_argument(
        "--bbb-group", default="high_confidence_BBB_permeable",
        help="Set empty to use --bbb-min-prob/--bbb-min-votes",
    )
    parser.add_argument("--bbb-min-prob", type=float, default=0.6)
    parser.add_argument("--bbb-min-votes", type=int, default=4)
    parser.add_argument("--smiles-column", default="model_smiles")
    parser.add_argument("--allow-canonical-smiles-fallback", action="store_true")
    parser.add_argument("--exclude-black-box-warning", action="store_true")
    parser.add_argument("--exclude-withdrawn", action="store_true")
    parser.add_argument("--exclude-multicomponent", action="store_true")
    parser.add_argument("--exclude-no-carbon", action="store_true")
    parser.add_argument(
        "--deduplicate-by",
        choices=("dta_smiles", "parent_structure_id", "chembl_id", "none"),
        default="dta_smiles",
    )
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--expected-candidates", type=int, default=0)

    parser.add_argument(
        "--ue-model-root", "--model-root", dest="model_root", type=Path,
        default=DEFAULT_UE_MODEL_ROOT,
    )
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--scenario", choices=ALLOWED_SCENARIOS, default="cold_target")
    parser.add_argument("--seeds", default="1-5")
    parser.add_argument("--strong-pkd-threshold", type=float, default=7.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--ligand-cache-size", type=int, default=4096)
    parser.add_argument("--inference-seed", type=int, default=20260727)

    parser.add_argument("--embedding-root", type=Path, default=DEFAULT_EMBEDDING_ROOT)
    parser.add_argument(
        "--ue-feature-cache-root", "--feature-cache-root",
        dest="feature_cache_root", type=Path, default=DEFAULT_UE_FEATURE_CACHE_ROOT,
    )
    parser.add_argument("--feature-device", default="auto")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--feature-checkpoint-every", type=int, default=25)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--no-offline-first", action="store_true")
    parser.add_argument("--hf-endpoint", default="")
    parser.add_argument("--unimol-model-name", default="unimolv2")
    parser.add_argument("--unimol-model-size", default="1.1B")
    parser.add_argument("--unimol-batch-size", type=int, default=32)
    parser.add_argument("--esmc-model-name", default="esmc_600m")
    parser.add_argument("--esmc-max-sequence-length", type=int, default=4500)
    parser.add_argument(
        "--esm2-backend", choices=("transformers", "fair_esm"), default="transformers"
    )
    parser.add_argument("--esm2-model-name", default="facebook/esm2_t33_650M_UR50D")
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

    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=2000)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--features-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    args.max_phases = parse_integer_set(args.max_phases, 1, 4, "max_phases")
    args.seeds = parse_seeds(args.seeds)
    args.bp_seeds = parse_seeds(args.bp_seeds)
    if args.max_phases != (4,):
        raise ValueError("This formal screening pipeline requires exact --max-phases 4")
    if args.max_candidates < 0 or args.expected_candidates < 0:
        raise ValueError("Candidate counts must be >= 0")
    if not 0 <= args.bbb_min_prob <= 1 or args.bbb_min_votes < 0:
        raise ValueError("Invalid numeric BBB filter")
    if args.batch_size <= 0 or args.chunk_size <= 0:
        raise ValueError("Batch and chunk sizes must be > 0")
    if args.bp_batch_size <= 0 or args.bp_feature_cache_size <= 0:
        raise ValueError("BP-NET batch/cache sizes must be > 0")
    if args.chembl_page_size <= 0 or args.download_retries <= 0:
        raise ValueError("ChEMBL page size/retry count must be > 0")
    if not 0 < args.bbb_vote_threshold < 1:
        raise ValueError("--bbb-vote-threshold must be in (0, 1)")
    if not 0 < args.bbb_positive_ppv_target <= 1:
        raise ValueError("--bbb-positive-ppv-target must be in (0, 1]")
    if not 0 < args.bbb_negative_npv_target <= 1:
        raise ValueError("--bbb-negative-npv-target must be in (0, 1]")
    if args.bbb_min_threshold_samples <= 0 or args.bbb_threshold_grid_size < 2:
        raise ValueError("Invalid BBB threshold calibration settings")
    if args.bbb_min_positive_votes is not None and not (
        1 <= args.bbb_min_positive_votes <= len(args.bp_seeds)
    ):
        raise ValueError("--bbb-min-positive-votes exceeds selected BP seeds")
    if args.num_workers < 0 or args.ligand_cache_size <= 0:
        raise ValueError("Invalid DataLoader or cache size")
    if args.feature_checkpoint_every <= 0 or args.unimol_batch_size <= 0:
        raise ValueError("Feature checkpoint and batch sizes must be > 0")
    if args.esmc_max_sequence_length < NLRP3_EXPECTED_LENGTH:
        raise ValueError("ESMC max sequence length would truncate canonical NLRP3")
    if not 0 <= args.esm2_window_overlap < args.esm2_window_size:
        raise ValueError("Invalid ESM2 window overlap")
    if not 0 <= args.esm2_probability_threshold <= 1:
        raise ValueError("Invalid ESM2 probability threshold")
    if args.esm2_top_k_long_range < 0 or args.esm2_minimum_separation < 1:
        raise ValueError("Invalid ESM2 sparsification settings")
    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")

    embedding_root = args.embedding_root.resolve()
    for label, cache_root in (
        ("--ue-feature-cache-root", args.feature_cache_root.resolve()),
        ("--bp-feature-cache-root", args.bp_feature_cache_root.resolve()),
    ):
        try:
            cache_root.relative_to(embedding_root)
        except ValueError as exc:
            raise ValueError(f"{label} must be inside --embedding-root") from exc
    if args.feature_cache_root.resolve() == args.bp_feature_cache_root.resolve():
        raise ValueError("BP-NET and UE-AlignNet feature cache roots must differ")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_arguments(args)
    set_determinism(args.inference_seed)

    code_root = Path(__file__).resolve().parents[1]
    ue = import_project_module(
        "project_train_ue_alignnet",
        code_root / "training" / "train_ue_alignnet.py",
    )
    bp = import_project_module(
        "project_train_bp_net",
        code_root / "training" / "train_bp_net.py",
    )
    eg = import_project_module(
        "project_embedding_generation",
        code_root / "preprocessing" / "embedding_generation.py",
    )
    device = torch.device(ue.resolve_device(args.device))
    cache_root = args.feature_cache_root.resolve()
    args.embedding_root = args.embedding_root.resolve()
    args.data_root = args.data_root.resolve()
    args.chembl_root = args.chembl_root.resolve()
    args.chembl_small_molecule_csv = args.chembl_small_molecule_csv.resolve()
    args.bbb_output_dir = args.bbb_output_dir.resolve()
    args.bp_model_root = args.bp_model_root.resolve()
    args.model_root = args.model_root.resolve()
    args.bp_feature_cache_root = args.bp_feature_cache_root.resolve()
    embedding_args = build_embedding_args(args, eg)

    upstream_bbb_manifest: dict[str, Any] = {}
    if args.candidate_csv is None:
        if args.check_only:
            existing_bbb_candidate = args.bbb_output_dir / DEFAULT_BBB_CANDIDATE_CSV.name
            if not existing_bbb_candidate.is_file():
                raise FileNotFoundError(
                    "--check-only does not download or run BP-NET. The default BBB "
                    f"candidate file is absent: {existing_bbb_candidate}"
                )
            args.candidate_csv = existing_bbb_candidate
        else:
            print("\n[Stage 1/3] ChEMBL download and small-molecule preparation")
            small_molecule_csv, chembl_summary = prepare_chembl_small_molecules(args)
            print(
                f"[ChEMBL] retained={chembl_summary['small_molecule_records']:,}, "
                f"source={chembl_summary['source']}"
            )
            print("\n[Stage 2/3] BP-NET BBB-permeability screening")
            args.candidate_csv, upstream_bbb_manifest = run_bp_net_screening(
                args, bp, eg, embedding_args, device, small_molecule_csv
            )
    else:
        args.candidate_csv = args.candidate_csv.resolve()
        print(f"[Stages 1-2 skipped] Using existing BBB candidate CSV: {args.candidate_csv}")

    upstream_manifest_path = args.candidate_csv.parent / "bbb_screening_manifest.json"
    if not upstream_bbb_manifest and upstream_manifest_path.is_file():
        try:
            with upstream_manifest_path.open("r", encoding="utf-8") as stream:
                upstream_bbb_manifest = json.load(stream)
        except (OSError, json.JSONDecodeError):
            upstream_bbb_manifest = {}

    if args.bbb_only:
        print(f"BBB screening complete: {args.candidate_csv}")
        return 0

    print("\n[Stage 3/3] UE-AlignNet NLRP3 affinity screening")
    candidates, selection_summary = load_candidates(args)
    target_header, target_sequence = read_nlrp3_fasta(
        args.nlrp3_fasta,
        args.allow_target_mismatch,
        auto_download=not args.offline,
    )
    checkpoints, architecture, effective = inspect_checkpoints(args)
    training_feature_configurations = verify_training_feature_configurations(
        eg, embedding_args
    )

    ligand_entities = [
        eg.Entity("BindingDB", "ligand", row.drug_id, row.dta_smiles)
        for row in candidates[["drug_id", "dta_smiles"]].itertuples(index=False)
    ]
    target_entity = eg.Entity(
        "BindingDB", "protein", NLRP3_ENTITY_ID, eg.clean_sequence(target_sequence)
    )
    scans = {
        "unimol2": scan_features(
            eg, embedding_args, cache_root, "unimol2", ligand_entities, args.force_features
        ),
        "rdkit": scan_features(
            eg, embedding_args, cache_root, "rdkit", ligand_entities, args.force_features
        ),
        "esmc": scan_features(
            eg, embedding_args, cache_root, "esmc", [target_entity], args.force_features
        ),
        "esm2_contact_graph": scan_features(
            eg,
            embedding_args,
            cache_root,
            "esm2_contact_graph",
            [target_entity],
            args.force_features,
        ),
    }
    pending_counts = {feature: len(scan.pending) for feature, scan in scans.items()}
    print(
        f"[Preflight] candidates={len(candidates):,}, target=NLRP3/{len(target_sequence)} aa, "
        f"scenario={args.scenario}, seeds={list(args.seeds)}, device={device}"
    )
    print(f"[Preflight] pending features={pending_counts}")
    if args.check_only:
        if any(pending_counts.values()):
            print("Preflight failed: one or more feature artifacts are missing or invalid.")
            return 2
        print("Preflight passed: candidates, target, checkpoints, and features are valid.")
        return 0

    failures = generate_feature_artifacts(
        eg, embedding_args, cache_root, scans
    )
    failed_ids = {
        entity_id
        for feature in ("unimol2", "rdkit")
        for entity_id, row in scans[feature].rows.items()
        if row["status"] != "valid"
    }
    for feature in ("esmc", "esm2_contact_graph"):
        if scans[feature].rows[NLRP3_ENTITY_ID]["status"] != "valid":
            raise RuntimeError(f"NLRP3 {feature} generation failed")
    eligible = candidates.loc[~candidates["drug_id"].isin(failed_ids)].copy().reset_index(drop=True)
    if eligible.empty:
        raise RuntimeError("Every candidate failed UE-AlignNet feature generation")
    selection_summary["technical_feature_failures"] = len(failed_ids)
    selection_summary["eligible_candidates"] = len(eligible)
    selection_summary["eligible_candidate_fingerprint_sha256"] = candidate_fingerprint(eligible)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        clear_prediction_outputs(args.output_dir)
    if failures:
        failure_frame = pd.DataFrame(failures).merge(
            candidates[["drug_id", "screening_compound_id", "chembl_id", "dta_smiles"]],
            left_on="entity_id",
            right_on="drug_id",
            how="left",
        ).drop(columns="drug_id")
    else:
        failure_frame = pd.DataFrame(
            columns=["entity_id", "feature", "error", "screening_compound_id", "chembl_id", "dta_smiles"]
        )

    if args.features_only:
        write_audit_outputs(
            args.output_dir, candidates, failure_frame, selection_summary
        )
        print(
            f"Feature generation complete: eligible={len(eligible):,}, failures={len(failed_ids):,}"
        )
        return 0

    feature_configurations = {
        feature: eg.feature_configuration(embedding_args, "BindingDB", feature)
        for feature in ("unimol2", "rdkit", "esmc", "esm2_contact_graph")
    }
    run_payload = {
        "script_version": SCRIPT_VERSION,
        "candidate_fingerprint": selection_summary["eligible_candidate_fingerprint_sha256"],
        "target_accession": NLRP3_ENTITY_ID,
        "target_header": target_header,
        "target_sequence_sha256": sha256_text(target_sequence),
        "scenario": args.scenario,
        "seeds": list(args.seeds),
        "checkpoints": checkpoints,
        "architecture": architecture,
        "effective": effective,
        "feature_configurations": feature_configurations,
        "training_feature_configuration_checks": training_feature_configurations,
        "strong_pKd_threshold": args.strong_pkd_threshold,
        "upstream_bbb": {
            "candidate_csv_sha256": sha256_file(args.candidate_csv),
            "manifest_path": str(upstream_manifest_path)
            if upstream_manifest_path.is_file() else None,
            "model_architecture": upstream_bbb_manifest.get("model_architecture"),
            "selected_seeds": nested_value(
                upstream_bbb_manifest, "thresholds", "selected_seeds"
            ),
            "positive_threshold": nested_value(
                upstream_bbb_manifest, "thresholds", "positive", "threshold"
            ),
            "minimum_positive_votes": nested_value(
                upstream_bbb_manifest, "thresholds", "minimum_positive_votes"
            ),
        },
    }
    run_fingerprint = sha256_text(
        json.dumps(json_safe(run_payload), sort_keys=True, separators=(",", ":"))
    )
    manifest_path = args.output_dir / "run_manifest.json"
    existing_manifest: dict[str, Any] = {}
    if manifest_path.is_file() and not args.overwrite:
        with manifest_path.open("r", encoding="utf-8") as stream:
            existing_manifest = json.load(stream)
        if existing_manifest.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError(
                "Existing prediction output belongs to a different candidate/target/model run. "
                "Use a new --output-dir or explicitly pass --overwrite."
            )
    write_audit_outputs(
        args.output_dir, candidates, failure_frame, selection_summary
    )
    manifest = {
        **run_payload,
        "run_fingerprint": run_fingerprint,
        "candidate_csv": str(args.candidate_csv),
        "nlrp3_fasta": str(args.nlrp3_fasta),
        "feature_cache_root": str(cache_root),
        "output_dir": str(args.output_dir),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "completed": False,
    }
    atomic_write_json(manifest_path, manifest)

    stores = build_stores(
        ue,
        args,
        cache_root,
        architecture,
        effective,
        eligible["drug_id"].astype(str).tolist(),
    )
    parts_dir = args.output_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    for checkpoint in checkpoints:
        predict_checkpoint_chunks(
            ue,
            args,
            checkpoint,
            architecture,
            effective,
            eligible,
            stores,
            parts_dir,
            run_fingerprint,
            device,
        )

    predictions = collect_seed_predictions(
        eligible, parts_dir, args.seeds, args.chunk_size, run_fingerprint
    )
    results = build_results(eligible, predictions, args, target_sequence)
    atomic_write_csv(results, args.output_dir / "all_predictions.csv")
    atomic_write_csv(results.head(args.top_k), args.output_dir / "top_candidates.csv")
    atomic_write_csv(
        results.loc[
            results["binding_confidence_group"].eq("high_confidence_strong_binder")
        ].copy(),
        args.output_dir / "high_confidence_strong_binders.csv",
    )
    manifest.update({
        "completed": True,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "predicted_candidates": len(results),
        "high_confidence_strong_binders": int(
            results["binding_confidence_group"].eq("high_confidence_strong_binder").sum()
        ),
        "outputs": {
            "all_predictions": str(args.output_dir / "all_predictions.csv"),
            "top_candidates": str(args.output_dir / "top_candidates.csv"),
            "high_confidence_strong_binders": str(
                args.output_dir / "high_confidence_strong_binders.csv"
            ),
        },
    })
    atomic_write_json(manifest_path, manifest)
    print(
        f"Screening complete: candidates={len(results):,}, "
        f"top pKd={results['pred_pKd_mean'].max():.4f}, output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
