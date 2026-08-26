from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import chromadb
import numpy as np
import pandas as pd
import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

import ontology_aligner_runtime as core

from . import config


def safe_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-").lower()


def main_record_path(dataset: str, stage: str) -> Path:
    filename = {
        "oar": "oar_records.jsonl",
        "lcr": "lcr_records.jsonl",
        "final": "final_records.jsonl",
    }[stage]
    return config.MAIN_RUN_DIR / config.MAIN_DATASET_KEYS[dataset] / filename


def load_full_stage(stage: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for dataset in config.DATASET_ORDER:
        values = core.load_jsonl(main_record_path(dataset, stage))
        for value in values:
            value["dataset_key"] = dataset
        records.extend(values)
    if len(records) != config.FULL_SAMPLE_COUNT:
        raise ValueError(
            f"Main {stage} records contain {len(records)} rows; "
            f"expected {config.FULL_SAMPLE_COUNT}"
        )
    return records


def load_subset_frame() -> pd.DataFrame:
    frame = pd.read_excel(
        config.SUBSET_PATH, sheet_name="Samples", dtype=str
    ).fillna("")
    if len(frame) != config.SUBSET_SAMPLE_COUNT:
        raise ValueError(f"Ablation subset contains {len(frame)} rows")
    reverse_keys = {value: key for key, value in config.SUBSET_DATASET_KEYS.items()}
    unknown = set(frame["dataset_key"]) - set(reverse_keys)
    if unknown:
        raise ValueError(f"Ablation subset contains unknown dataset keys: {unknown}")
    frame["dataset_key"] = frame["dataset_key"].map(reverse_keys)
    counts = frame.groupby("dataset_key").size().to_dict()
    expected = {
        dataset: config.SUBSET_ROWS_PER_DATASET for dataset in config.DATASET_ORDER
    }
    if counts != expected:
        raise ValueError(f"Ablation subset dataset counts changed: {counts}")
    return frame


def load_subset_stage(stage: str) -> list[dict[str, Any]]:
    subset = load_subset_frame()
    output: list[dict[str, Any]] = []
    for dataset in config.DATASET_ORDER:
        source = {
            int(record["source_excel_row"]): record
            for record in core.load_jsonl(main_record_path(dataset, stage))
        }
        selected = subset[subset["dataset_key"] == dataset]
        for row in selected.to_dict(orient="records"):
            excel_row = int(row["source_excel_row"])
            if excel_row not in source:
                raise ValueError(f"{dataset}: source_excel_row={excel_row} is missing")
            record = dict(source[excel_row])
            if int(row["source_sample_id"]) != int(record["source_sample_id"]) + 1:
                raise ValueError(
                    f"{dataset}: subset/main sample identity mismatch at row {excel_row}"
                )
            if str(row["Raw_Phenotype_Names"]) != str(record["query"]):
                raise ValueError(f"{dataset}: query mismatch at row {excel_row}")
            record["dataset_key"] = dataset
            output.append(record)
    if len(output) != config.SUBSET_SAMPLE_COUNT:
        raise ValueError(f"Mapped subset contains {len(output)} rows")
    return output


def group_by_dataset(
    records: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped = {dataset: [] for dataset in config.DATASET_ORDER}
    for record in records:
        grouped[str(record["dataset_key"])].append(record)
    return grouped


def evaluate_predictions(
    records: Sequence[dict[str, Any]], method: str | None = None
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        if method is not None:
            item["method"] = method
        item["correct"] = item["predicted_id"] in set(item["accepted_ids"])
        output.append(item)
    return output


def prefix_candidates(
    records: Sequence[dict[str, Any]], candidate_k: int
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        item["candidate_ids"] = list(record["candidate_ids"][:candidate_k])
        item["candidate_scores"] = list(record["candidate_scores"][:candidate_k])
        item["candidate_matched_surfaces"] = list(
            record["candidate_matched_surfaces"][:candidate_k]
        )
        item["predicted_id"] = item["candidate_ids"][0]
        item["correct"] = item["predicted_id"] in set(item["accepted_ids"])
        output.append(item)
    return output


def raw_collection(spec: config.BackboneSpec) -> Any:
    client = chromadb.PersistentClient(path=str(config.RAW_CHROMA_PATH))
    collection = client.get_collection(spec.collection_name)
    metadata = collection.metadata or {}
    if collection.count() != 44_814:
        raise ValueError(f"{spec.collection_name}: unexpected surface count")
    if int(metadata.get("embedding_dimensions", 0)) != spec.dimension:
        raise ValueError(f"{spec.collection_name}: dimension metadata mismatch")
    if str(metadata.get("model_revision")) != spec.revision:
        raise ValueError(f"{spec.collection_name}: revision metadata mismatch")
    if str(metadata.get("pooling")) != spec.pooling:
        raise ValueError(f"{spec.collection_name}: pooling metadata mismatch")
    return collection


def load_collection_matrix(
    collection: Any, dimension: int, chunk_size: int = 512
) -> tuple[np.ndarray, list[str], list[dict[str, Any]], list[str]]:
    count = collection.count()
    matrix = np.empty((count, dimension), dtype=np.float32)
    documents: list[str] = [""] * count
    metadatas: list[dict[str, Any]] = [{} for _ in range(count)]
    ids = [f"surface_{index:07d}" for index in range(count)]
    for start in range(0, count, chunk_size):
        requested = ids[start : start + chunk_size]
        payload = collection.get(
            ids=requested, include=["embeddings", "documents", "metadatas"]
        )
        positions = {str(value): index for index, value in enumerate(payload["ids"])}
        for offset, surface_id in enumerate(requested):
            position = positions[surface_id]
            target = start + offset
            matrix[target] = np.asarray(
                payload["embeddings"][position], dtype=np.float32
            )
            documents[target] = str(payload["documents"][position])
            metadatas[target] = dict(payload["metadatas"][position] or {})
    return matrix, documents, metadatas, ids


class QueryEmbeddingCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS embeddings "
            "(cache_key TEXT PRIMARY KEY, vector BLOB NOT NULL, dimension INTEGER NOT NULL, "
            "source TEXT NOT NULL)"
        )
        self.connection.commit()

    @staticmethod
    def key(model: str, text: str) -> str:
        return hashlib.sha256(f"{model}\0{text}".encode()).hexdigest()

    def get(self, model: str, text: str, dimension: int) -> np.ndarray | None:
        row = self.connection.execute(
            "SELECT vector, dimension FROM embeddings WHERE cache_key = ?",
            (self.key(model, text),),
        ).fetchone()
        if row is None or int(row[1]) != dimension:
            return None
        vector = np.frombuffer(row[0], dtype=np.float32).copy()
        return vector if vector.shape == (dimension,) else None

    def put_many(
        self, model: str, texts: Sequence[str], vectors: np.ndarray, source: str
    ) -> None:
        self.connection.executemany(
            "INSERT OR REPLACE INTO embeddings(cache_key, vector, dimension, source) "
            "VALUES (?, ?, ?, ?)",
            [
                (
                    self.key(model, text),
                    np.asarray(vector, dtype=np.float32).tobytes(),
                    int(vectors.shape[1]),
                    source,
                )
                for text, vector in zip(texts, vectors)
            ],
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


class PinnedEmbeddingBackend:
    def __init__(self, spec: config.BackboneSpec):
        self.spec = spec
        self.settings = core.load_config().get(spec.config_section or "embedding", {})
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.thread_local = threading.local()
        self.local_model: Any = None
        self.tokenizer: Any = None
        self.transformer: Any = None
        if spec.backend == "sentence_transformers":
            self.local_model = SentenceTransformer(
                spec.model_name,
                revision=spec.revision,
                device=self.device,
                local_files_only=True,
            )
        elif spec.backend == "transformers_mean_pooling":
            self.tokenizer = AutoTokenizer.from_pretrained(
                spec.model_name, revision=spec.revision, local_files_only=True
            )
            self.transformer = AutoModel.from_pretrained(
                spec.model_name, revision=spec.revision, local_files_only=True
            ).to(self.device)
            self.transformer.eval()
        elif spec.backend != "openai":
            raise ValueError(f"Unsupported embedding backend: {spec.backend}")

    def _client(self) -> OpenAI:
        if not hasattr(self.thread_local, "client"):
            kwargs: dict[str, Any] = {
                "api_key": self.settings["api_key"],
                "max_retries": 0,
            }
            if self.settings.get("base_url"):
                kwargs["base_url"] = self.settings["base_url"]
            self.thread_local.client = OpenAI(**kwargs)
        return self.thread_local.client

    def _remote_batch(self, texts: Sequence[str]) -> np.ndarray:
        response = self._client().embeddings.create(
            model=self.spec.model_name, input=list(texts)
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        return np.asarray([item.embedding for item in ordered], dtype=np.float32)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        values = [str(value) for value in texts]
        if self.spec.backend == "openai":
            batches = [
                values[index : index + self.spec.batch_size]
                for index in range(0, len(values), self.spec.batch_size)
            ]
            concurrency = int(self.settings.get("max_concurrency", 10))
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                parts = list(executor.map(self._remote_batch, batches))
            vectors = np.vstack(parts)
        elif self.local_model is not None:
            vectors = self.local_model.encode(
                values,
                batch_size=self.spec.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=False,
            ).astype(np.float32, copy=False)
        else:
            parts = []
            with torch.inference_mode():
                for start in range(0, len(values), self.spec.batch_size):
                    tokens = self.tokenizer(
                        values[start : start + self.spec.batch_size],
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    ).to(self.device)
                    hidden = self.transformer(**tokens).last_hidden_state
                    mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)
                    parts.append(pooled.float().cpu().numpy())
            vectors = np.vstack(parts)
        if vectors.shape != (len(values), self.spec.dimension):
            raise ValueError(
                f"{self.spec.key}: unexpected embedding shape {vectors.shape}"
            )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return np.asarray(vectors / np.maximum(norms, 1e-12), dtype=np.float32)


def resolve_query_embeddings(
    spec: config.BackboneSpec, texts: Sequence[str]
) -> tuple[np.ndarray, dict[str, int]]:
    values = [str(value) for value in texts]
    unique = list(dict.fromkeys(values))
    cache = QueryEmbeddingCache(config.QUERY_EMBEDDING_CACHE)
    vectors: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for text in unique:
        vector = cache.get(spec.model_name, text, spec.dimension)
        if vector is None:
            missing.append(text)
        else:
            vectors[text] = vector
    source_hits = 0
    if spec.key == "text_embedding_3_large" and missing:
        source = sqlite3.connect(
            f"file:{core.SOURCE_EMBEDDING_CACHE}?mode=ro", uri=True
        )
        still_missing = []
        for text in missing:
            row = source.execute(
                "SELECT vector, dimension FROM embeddings WHERE cache_key = ?",
                (cache.key(spec.model_name, text),),
            ).fetchone()
            if row is None or int(row[1]) != spec.dimension:
                still_missing.append(text)
                continue
            vector = np.frombuffer(row[0], dtype=np.float32).copy()
            if vector.shape != (spec.dimension,):
                still_missing.append(text)
                continue
            vectors[text] = vector
            cache.put_many(
                spec.model_name, [text], vector.reshape(1, -1), "main_embedding_cache"
            )
            source_hits += 1
        source.close()
        missing = still_missing
    generated = 0
    if missing:
        backend = PinnedEmbeddingBackend(spec)
        computed = backend.encode(missing)
        cache.put_many(spec.model_name, missing, computed, spec.backend)
        vectors.update(zip(missing, computed))
        generated = len(missing)
        del backend
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    cache.close()
    matrix = np.vstack([vectors[text] for text in values]).astype(np.float32)
    return matrix, {
        "rows": len(values),
        "unique": len(unique),
        "cache_hits": len(unique) - source_hits - generated,
        "source_cache_hits": source_hits,
        "generated": generated,
    }


def retrieve_records(
    samples: Sequence[dict[str, Any]], collection: Any, vectors: np.ndarray, method: str
) -> list[dict[str, Any]]:
    candidates = core.retrieve_many(collection, vectors, top_k=config.RETRIEVAL_TOP_K)
    return evaluate_predictions(core.retrieval_records(samples, candidates, method))


def dynamic_prompt(template: str, candidate_k: int) -> str:
    if candidate_k == 20:
        return template
    return (
        template.replace("20 candidate", f"{candidate_k} candidate")
        .replace("all 20 candidates", f"all {candidate_k} candidates")
        .replace("20 candidate HPO", f"{candidate_k} candidate HPO")
    )


def prompt_hashes(candidate_k: int) -> dict[str, str]:
    lcr = dynamic_prompt(core.USER_PROMPT_TEMPLATE, candidate_k)
    hgr = dynamic_prompt(core.HPO_GRAPH_RERANK_PROMPT, candidate_k)
    return {
        "lcr_system": hashlib.sha256(core.SYSTEM_PROMPT.encode()).hexdigest(),
        "lcr_user_template": hashlib.sha256(lcr.encode()).hexdigest(),
        "hgr_system": hashlib.sha256(core.HPO_GRAPH_SYSTEM_PROMPT.encode()).hexdigest(),
        "hgr_user_template": hashlib.sha256(hgr.encode()).hexdigest(),
    }


def build_lcr_task(
    record: dict[str, Any], terms: dict[str, Any], stage: str, model: str
) -> dict[str, Any]:
    candidate_ids = list(record["candidate_ids"])
    candidate_k = len(candidate_ids)
    payload = json.dumps(
        {
            "original_text": record["query"],
            "candidates": core.candidate_payloads(candidate_ids, terms),
        },
        ensure_ascii=False,
        indent=2,
    )
    user_prompt = dynamic_prompt(core.USER_PROMPT_TEMPLATE, candidate_k).format(
        input_payload=payload
    )
    identity = {
        "stage": stage,
        "model": model,
        "request_options": core.llm_request_options(model),
        "temperature": core.LLM_TEMPERATURE,
        "system_prompt_sha256": hashlib.sha256(core.SYSTEM_PROMPT.encode()).hexdigest(),
        "user_prompt_sha256": hashlib.sha256(user_prompt.encode()).hexdigest(),
        "candidate_ids": candidate_ids,
    }
    return {
        "key": core.stable_hash(identity),
        "stage": stage,
        "model": model,
        "system_prompt": core.SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "allowed_ids": candidate_ids,
        "allow_no_match": True,
    }


def build_hgr_task(
    record: dict[str, Any], terms: dict[str, Any], stage: str, model: str
) -> dict[str, Any]:
    candidate_ids = list(record["ranking"])
    candidate_k = len(candidate_ids)
    payload = json.dumps(
        {
            "original_text": record["query"],
            "candidates": core.candidate_payloads(candidate_ids, terms),
            "hpo_graph_context": record["hpo_graph_context"],
        },
        ensure_ascii=False,
        indent=2,
    )
    user_prompt = dynamic_prompt(core.HPO_GRAPH_RERANK_PROMPT, candidate_k).format(
        input_payload=payload
    )
    identity = {
        "stage": stage,
        "model": model,
        "request_options": core.llm_request_options(model),
        "temperature": core.LLM_TEMPERATURE,
        "system_prompt_sha256": hashlib.sha256(
            core.HPO_GRAPH_SYSTEM_PROMPT.encode()
        ).hexdigest(),
        "user_prompt_sha256": hashlib.sha256(user_prompt.encode()).hexdigest(),
        "candidate_ids": candidate_ids,
    }
    return {
        "key": core.stable_hash(identity),
        "stage": stage,
        "model": model,
        "system_prompt": core.HPO_GRAPH_SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "allowed_ids": candidate_ids,
        "allow_no_match": False,
    }


def llm_cache_path(model: str, phase: str) -> Path:
    return config.LLM_CACHE_DIR / f"{safe_tag(model)}_{phase}.jsonl"


def run_lcr(
    records: Sequence[dict[str, Any]], config_key: str, stage: str
) -> list[dict[str, Any]]:
    settings = core.load_config()[config_key]
    model = str(settings["model"])
    terms = core.load_ontology()
    tasks = [build_lcr_task(record, terms, stage, model) for record in records]
    runner = core.LLMTaskRunner(settings, model, config.REQUESTS_PER_MINUTE)
    responses = core.run_unique_tasks(
        tasks,
        llm_cache_path(model, "lcr"),
        runner,
        config.MAX_CONCURRENCY,
    )
    assembled = core.assemble_lcr_records(records, tasks, responses, "LCR")
    return assembled


def run_hgr(
    records: Sequence[dict[str, Any]], config_key: str, stage: str
) -> list[dict[str, Any]]:
    settings = core.load_config()[config_key]
    model = str(settings["model"])
    terms = core.load_ontology()
    graph = core.load_hpo_graph()
    triggered: list[dict[str, Any]] = []
    prepared: list[dict[str, Any]] = []
    for source in records:
        record = dict(source)
        context = core.build_hpo_graph_context(
            record["ranking"], record["candidate_ids"][0], terms, graph
        )
        record["hpo_graph_context"] = context
        record["graph_triggered"] = bool(context)
        prepared.append(record)
        if context:
            triggered.append(record)
    tasks = [build_hgr_task(record, terms, stage, model) for record in triggered]
    runner = core.LLMTaskRunner(settings, model, config.REQUESTS_PER_MINUTE)
    responses = core.run_unique_tasks(
        tasks,
        llm_cache_path(model, "hgr"),
        runner,
        config.MAX_CONCURRENCY,
    )
    response_by_sample = {
        record["sample_id"]: (task, responses[task["key"]])
        for record, task in zip(triggered, tasks)
    }
    final: list[dict[str, Any]] = []
    for record in prepared:
        output = dict(record)
        output["method"] = "OntologyAligner"
        output["initial_ranking"] = list(record["ranking"])
        output["initial_predicted_id"] = record["predicted_id"]
        output["graph_rerank_applied"] = record["sample_id"] in response_by_sample
        if output["graph_rerank_applied"]:
            task, response = response_by_sample[record["sample_id"]]
            ranking = list(response["ranking"])
            if response["no_match"] or not ranking:
                raise ValueError(f"{stage}: HGR produced No Match")
            output["ranking"] = ranking
            output["no_match"] = False
            output["predicted_id"] = ranking[0]
            output["hgr_response_key"] = task["key"]
            output["hgr_attempts"] = response["attempts"]
            output["hgr_api_seconds"] = response["api_seconds"]
            output["hgr_prompt_tokens"] = response["prompt_tokens"]
            output["hgr_completion_tokens"] = response["completion_tokens"]
            output["hgr_total_tokens"] = response["total_tokens"]
            output["hgr_raw_response"] = response["raw_response"]
            output["hgr_ranking_repair"] = response.get("ranking_repair", {})
        else:
            output.update(
                {
                    "hgr_response_key": "",
                    "hgr_attempts": 0,
                    "hgr_api_seconds": 0.0,
                    "hgr_prompt_tokens": 0,
                    "hgr_completion_tokens": 0,
                    "hgr_total_tokens": 0,
                    "hgr_raw_response": "",
                    "hgr_ranking_repair": {},
                }
            )
        output["correct"] = output["predicted_id"] in set(output["accepted_ids"])
        final.append(output)
    return final


def write_checkpoint(path: Path, records: Sequence[dict[str, Any]]) -> None:
    core.write_jsonl(path, records)


def read_complete_checkpoint(path: Path, expected_rows: int) -> list[dict[str, Any]] | None:
    records = core.load_jsonl(path)
    if not records:
        return None
    if len(records) != expected_rows:
        raise ValueError(f"Incomplete checkpoint {path}: {len(records)}/{expected_rows}")
    return records
