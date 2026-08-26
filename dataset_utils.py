from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
CURRENT_HPO = PROJECT_ROOT / "HPO" / "hp260623.obo"
NO_MATCH = "No Match"
ACCEPTED_HPO_IDS_COLUMN = "Accepted_HPO_ids"


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    workbook: Path
    sheet: str
    inference_columns: tuple[str, ...]


DATASETS = {
    "fgdd": DatasetConfig(
        key="fgdd",
        workbook=PROJECT_ROOT / "Dataset" / "FGDD" / "FGDD_Phenotype_Testset.xlsx",
        sheet="FGDD",
        inference_columns=("Raw_Phenotype_Names", "patient_ids", "pmids"),
    ),
    "csc": DatasetConfig(
        key="csc",
        workbook=PROJECT_ROOT / "Dataset" / "CSC" / "CSC.xlsx",
        sheet="CSC",
        inference_columns=("Raw_Phenotype_Names", "Patient_ID"),
    ),
    "genereviews-10": DatasetConfig(
        key="genereviews-10",
        workbook=PROJECT_ROOT / "Dataset" / "GeneReviews-10" / "GeneReviews-10.xlsx",
        sheet="GeneReviews-10",
        inference_columns=("Raw_Phenotype_Names", "Patient_ID"),
    ),
    "bc8_t3": DatasetConfig(
        key="bc8_t3",
        workbook=PROJECT_ROOT / "Dataset" / "BC8_T3" / "BC8_T3.xlsx",
        sheet="BC8_T3",
        inference_columns=("Raw_Phenotype_Names", "Patient_ID"),
    ),
    "gsc2017": DatasetConfig(
        key="gsc2017",
        workbook=PROJECT_ROOT / "Dataset" / "GSC2017" / "GSC2017.xlsx",
        sheet="GSC2017",
        inference_columns=("Raw_Phenotype_Names", "Patient_ID"),
    ),
    "gsc2024": DatasetConfig(
        key="gsc2024",
        workbook=PROJECT_ROOT / "Dataset" / "GSC2024" / "GSC2024.xlsx",
        sheet="GSC2024",
        inference_columns=("Raw_Phenotype_Names", "Patient_ID"),
    ),
    "id-68": DatasetConfig(
        key="id-68",
        workbook=PROJECT_ROOT / "Dataset" / "ID-68" / "ID-68.xlsx",
        sheet="ID-68",
        inference_columns=("Raw_Phenotype_Names", "Patient_ID"),
    ),
}
DATASET_CHOICES = tuple(DATASETS)


@dataclass(frozen=True)
class HpoTerm:
    hpo_id: str
    alt_ids: tuple[str, ...]
    obsolete: bool
    replaced_by: tuple[str, ...]


def parse_obo(path: Path) -> dict[str, HpoTerm]:
    terms: dict[str, HpoTerm] = {}
    stanza: dict[str, list[str]] = {}
    in_term = False

    def flush() -> None:
        if not in_term or "id" not in stanza:
            return
        hpo_id = stanza["id"][0]
        terms[hpo_id] = HpoTerm(
            hpo_id=hpo_id,
            alt_ids=tuple(stanza.get("alt_id", [])),
            obsolete=stanza.get("is_obsolete", ["false"])[0].lower() == "true",
            replaced_by=tuple(value.strip() for value in stanza.get("replaced_by", [])),
        )

    with path.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            if line == "[Term]":
                flush()
                stanza = {}
                in_term = True
                continue
            if line.startswith("["):
                flush()
                stanza = {}
                in_term = False
                continue
            if not in_term or not line or line.startswith("!") or ": " not in line:
                continue
            key, value = line.split(": ", 1)
            stanza.setdefault(key, []).append(value)
    flush()
    return terms


class HpoCanonicalizer:
    def __init__(self, obo_path: Path = CURRENT_HPO):
        self.terms = parse_obo(obo_path)
        self.alt_to_primary = {
            alt_id: term.hpo_id
            for term in self.terms.values()
            for alt_id in term.alt_ids
        }

    def canonicalize(self, raw_id: Any) -> tuple[str | None, str]:
        if raw_id is None or pd.isna(raw_id):
            return None, "no_match"
        value = str(raw_id).strip()
        if not value or value.lower() in {"none", "null", "nan", NO_MATCH.lower()}:
            return None, "no_match"
        if not re.fullmatch(r"HP:\d{7}", value):
            return None, "invalid_id"
        primary = self.alt_to_primary.get(value, value)
        term = self.terms.get(primary)
        if term is None:
            return None, "invalid_id"
        visited: set[str] = set()
        while term.obsolete and len(term.replaced_by) == 1 and term.hpo_id not in visited:
            visited.add(term.hpo_id)
            primary = self.alt_to_primary.get(term.replaced_by[0], term.replaced_by[0])
            term = self.terms.get(primary)
            if term is None:
                return None, "invalid_id"
        if term.obsolete:
            return None, "obsolete_unresolved"
        return primary, "valid"


def dataset_config(dataset: str = "fgdd") -> DatasetConfig:
    try:
        return DATASETS[dataset.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {dataset}") from exc


def load_gold_labels(limit: int | None = None, dataset: str = "fgdd") -> pd.DataFrame:
    config = dataset_config(dataset)
    frame = pd.read_excel(
        config.workbook,
        sheet_name=config.sheet,
        usecols=["HPO_id", ACCEPTED_HPO_IDS_COLUMN],
        dtype={"HPO_id": str, ACCEPTED_HPO_IDS_COLUMN: str},
    )
    if limit is not None:
        frame = frame.iloc[:limit].copy()
    frame = frame.reset_index(drop=True)
    frame.insert(0, "row_index", frame.index)
    return frame


def canonical_gold_ids(
    raw_gold: Any,
    canonicalizer: HpoCanonicalizer,
    raw_accepted: Any = None,
) -> tuple[str, ...]:
    required_gold_ids = re.findall(r"HP:\d{7}", str(raw_gold))
    accepted_is_missing = (
        raw_accepted is None
        or (not isinstance(raw_accepted, (list, tuple)) and pd.isna(raw_accepted))
        or not str(raw_accepted).strip()
    )
    if accepted_is_missing:
        raw_ids = required_gold_ids
        separator_only = re.sub(r"HP:\d{7}", "", str(raw_gold)).strip(" ,;|")
        if not raw_ids or separator_only:
            raise ValueError(f"Gold HPO ID is not canonicalizable: {raw_gold}")
    elif isinstance(raw_accepted, (list, tuple)):
        raw_ids = [str(value).strip() for value in raw_accepted]
    else:
        try:
            payload = json.loads(str(raw_accepted))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Accepted HPO IDs are not valid JSON: {raw_accepted}") from exc
        if not isinstance(payload, list):
            raise ValueError(f"Accepted HPO IDs must be a JSON array: {raw_accepted}")
        raw_ids = [str(value).strip() for value in payload]

    missing = [gold_id for gold_id in required_gold_ids if gold_id not in raw_ids]
    if missing:
        raise ValueError(f"Current gold HPO ID is missing from accepted IDs: {missing[0]}")
    invalid = [raw_id for raw_id in raw_ids if not re.fullmatch(r"HP:\d{7}", raw_id)]
    if invalid:
        raise ValueError(f"Gold HPO ID is not canonicalizable: {invalid[0]}")

    canonical_ids: list[str] = []
    for raw_id in raw_ids:
        canonical, status = canonicalizer.canonicalize(raw_id)
        if status == "obsolete_unresolved":
            canonical = raw_id
        elif status != "valid" or not canonical:
            raise ValueError(f"Gold HPO ID is not canonicalizable: {raw_id}")
        if canonical not in canonical_ids:
            canonical_ids.append(canonical)
    return tuple(canonical_ids)
