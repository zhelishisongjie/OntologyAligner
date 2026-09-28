from __future__ import annotations

import ast
import json
import re
import sqlite3
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from openai import OpenAI
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parent
HPO_JSONL_PATH = ROOT / "HPO" / "hp260623.jsonl"
HPO_GRAPH_PATH = ROOT / "HPO" / "hp260623.json"
CONFIG_PATH = ROOT / "LLM_config.json"
RERANK_NOTEBOOK_PATH = ROOT / "02_run_LLM_rerank.ipynb"
from ontology_aligner_oar import DEFAULT_OAR_CHROMA_PATH, OAR_COLLECTION_NAME

OAR_CHROMA_PATH = ROOT / DEFAULT_OAR_CHROMA_PATH

EMBEDDING_MODEL = "text-embedding-3-large"
LLM_CANDIDATE_COUNT = 20
HPO_GRAPH_PROBE_TOP_K = 3
HPO_MAX_ANCESTOR_DISTANCE = 2
LLM_TEMPERATURE = 0.0
REQUESTS_PER_MINUTE = 200
MAX_ATTEMPTS = 3
RETRIEVAL_BATCH_SIZE = 64
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 42

PROMPT_NAMES = (
    "SYSTEM_PROMPT",
    "USER_PROMPT_TEMPLATE",
    "HPO_GRAPH_SYSTEM_PROMPT",
    "HPO_GRAPH_RERANK_PROMPT",
)


def load_notebook_prompts(path: Path = RERANK_NOTEBOOK_PATH) -> dict[str, str]:
    notebook = json.loads(path.read_text(encoding="utf-8-sig"))
    code = "\n\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    values: dict[str, str] = {}
    for node in ast.parse(code).body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id in PROMPT_NAMES:
                value = ast.literal_eval(node.value)
                if not isinstance(value, str):
                    raise TypeError(f"Notebook prompt {target.id} is not a string literal")
                values[target.id] = value
    missing = set(PROMPT_NAMES) - set(values)
    if missing:
        raise ValueError(f"Notebook is missing prompt assignments: {sorted(missing)}")
    return values


NOTEBOOK_PROMPTS = load_notebook_prompts()
SYSTEM_PROMPT = NOTEBOOK_PROMPTS["SYSTEM_PROMPT"]
USER_PROMPT_TEMPLATE = NOTEBOOK_PROMPTS["USER_PROMPT_TEMPLATE"]
HPO_GRAPH_SYSTEM_PROMPT = NOTEBOOK_PROMPTS["HPO_GRAPH_SYSTEM_PROMPT"]
HPO_GRAPH_RERANK_PROMPT = NOTEBOOK_PROMPTS["HPO_GRAPH_RERANK_PROMPT"]


@dataclass(frozen=True)
class OntologyTerm:
    ontology_id: str
    name: str
    synonyms: tuple[str, ...]
    definition: str


@dataclass(frozen=True)
class HPOGraphIndex:
    parents: dict[str, frozenset[str]]
    names: dict[str, str]
    definitions: dict[str, str]


class GlobalRequestRateLimiter:
    def __init__(self, requests_per_minute: int):
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self.interval_seconds = 60.0 / requests_per_minute
        self._condition = threading.Condition()
        self._next_request_at = 0.0
        self._blocked_until = 0.0

    def acquire(self) -> None:
        with self._condition:
            while True:
                now = time.monotonic()
                ready_at = max(self._next_request_at, self._blocked_until)
                wait_seconds = ready_at - now
                if wait_seconds <= 0:
                    self._next_request_at = now + self.interval_seconds
                    return
                self._condition.wait(timeout=wait_seconds)

    def pause(self, seconds: float) -> None:
        with self._condition:
            self._blocked_until = max(
                self._blocked_until, time.monotonic() + float(seconds)
            )
            self._next_request_at = max(self._next_request_at, self._blocked_until)
            self._condition.notify_all()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8-sig",
    )
    temporary.replace(path)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(path)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8-sig", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_ontology() -> dict[str, OntologyTerm]:
    terms: dict[str, OntologyTerm] = {}
    with HPO_JSONL_PATH.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("is_obsolete"):
                continue
            ontology_id = str(item["HPOid"])
            synonyms = tuple(str(entry["text"]) for entry in item.get("synonyms", []))
            terms[ontology_id] = OntologyTerm(
                ontology_id=ontology_id,
                name=str(item["name"]),
                synonyms=tuple(dict.fromkeys(text for text in synonyms if text)),
                definition=str(item.get("definition") or ""),
            )
    return terms


def normalize_hpo_uri(value: str) -> str:
    local_id = str(value).rsplit("/", 1)[-1]
    return local_id.replace("HP_", "HP:", 1)


def load_hpo_graph() -> HPOGraphIndex:
    graph = json.loads(HPO_GRAPH_PATH.read_text(encoding="utf-8-sig"))["graphs"][0]
    parents: dict[str, set[str]] = defaultdict(set)
    names: dict[str, str] = {}
    definitions: dict[str, str] = {}
    for node in graph.get("nodes", []):
        ontology_id = normalize_hpo_uri(node["id"])
        meta = node.get("meta") or {}
        definition_meta = meta.get("definition") or {}
        names[ontology_id] = str(node.get("lbl") or ontology_id)
        definitions[ontology_id] = str(definition_meta.get("val") or "")
    for edge in graph.get("edges", []):
        if edge.get("pred") == "is_a":
            parents[normalize_hpo_uri(edge["sub"])].add(normalize_hpo_uri(edge["obj"]))
    return HPOGraphIndex(
        parents={key: frozenset(value) for key, value in parents.items()},
        names=names,
        definitions=definitions,
    )


def find_is_a_path(
    child_id: str, ancestor_id: str, graph: HPOGraphIndex, max_edges: int
) -> list[str]:
    queue = deque([(child_id, [child_id])])
    visited = {child_id}
    while queue:
        current_id, path = queue.popleft()
        if len(path) - 1 >= max_edges:
            continue
        for parent_id in sorted(graph.parents.get(current_id, ())):
            next_path = [*path, parent_id]
            if parent_id == ancestor_id:
                return next_path
            if parent_id not in visited:
                visited.add(parent_id)
                queue.append((parent_id, next_path))
    return []


def graph_concept_payload(
    ontology_id: str,
    terms: dict[str, OntologyTerm],
    graph: HPOGraphIndex,
) -> dict[str, str]:
    term = terms.get(ontology_id)
    name = term.name if term else graph.names.get(ontology_id, ontology_id)
    definition = term.definition if term else graph.definitions.get(ontology_id, "")
    return {
        "id": ontology_id,
        "preferred_name": name,
        "definition": definition or "No definition provided in this HPO release.",
    }


def is_a_relation_payload(
    child_id: str,
    parent_id: str,
    terms: dict[str, OntologyTerm],
    graph: HPOGraphIndex,
) -> dict[str, str]:
    child = graph_concept_payload(child_id, terms, graph)
    parent = graph_concept_payload(parent_id, terms, graph)
    return {
        "subject_id": child_id,
        "subject_name": child["preferred_name"],
        "predicate": "is_a",
        "object_id": parent_id,
        "object_name": parent["preferred_name"],
    }


def build_hpo_graph_context(
    llm_ranking: Sequence[str],
    dense_top1_id: str,
    terms: dict[str, OntologyTerm],
    graph: HPOGraphIndex,
) -> list[dict[str, Any]]:
    if not llm_ranking or llm_ranking[0] == dense_top1_id:
        return []
    probe_ids = list(
        dict.fromkeys([*llm_ranking[:HPO_GRAPH_PROBE_TOP_K], dense_top1_id])
    )
    contexts: list[dict[str, Any]] = []
    for left_id, right_id in combinations(probe_ids, 2):
        if dense_top1_id not in {left_id, right_id}:
            continue
        path = find_is_a_path(left_id, right_id, graph, HPO_MAX_ANCESTOR_DISTANCE)
        if not path:
            path = find_is_a_path(
                right_id, left_id, graph, HPO_MAX_ANCESTOR_DISTANCE
            )
        if path:
            descendant = graph_concept_payload(path[0], terms, graph)
            ancestor = graph_concept_payload(path[-1], terms, graph)
            contexts.append(
                {
                    "related_concepts": [descendant, ancestor],
                    "relationship_explanation": (
                        f"{descendant['preferred_name']} ({descendant['id']}) is a more "
                        f"specific descendant of {ancestor['preferred_name']} "
                        f"({ancestor['id']}) in HPO. The relationship is represented by "
                        "the following is_a path."
                    ),
                    "is_a_relations": [
                        is_a_relation_payload(child_id, parent_id, terms, graph)
                        for child_id, parent_id in zip(path, path[1:])
                    ],
                    "concept_definitions": [
                        graph_concept_payload(ontology_id, terms, graph)
                        for ontology_id in path
                    ],
                }
            )
            continue
        shared_parent_ids = sorted(
            graph.parents.get(left_id, frozenset())
            & graph.parents.get(right_id, frozenset())
        )
        if not shared_parent_ids:
            continue
        left = graph_concept_payload(left_id, terms, graph)
        right = graph_concept_payload(right_id, terms, graph)
        shared_names = [
            graph_concept_payload(parent_id, terms, graph)["preferred_name"]
            for parent_id in shared_parent_ids
        ]
        relation_nodes = [left_id, right_id, *shared_parent_ids]
        contexts.append(
            {
                "related_concepts": [left, right],
                "relationship_explanation": (
                    f"{left['preferred_name']} ({left_id}) and {right['preferred_name']} "
                    f"({right_id}) are sibling HPO concepts because they share the "
                    f"following direct parent(s): {', '.join(shared_names)}."
                ),
                "is_a_relations": [
                    *[
                        is_a_relation_payload(left_id, parent_id, terms, graph)
                        for parent_id in shared_parent_ids
                    ],
                    *[
                        is_a_relation_payload(right_id, parent_id, terms, graph)
                        for parent_id in shared_parent_ids
                    ],
                ],
                "concept_definitions": [
                    graph_concept_payload(ontology_id, terms, graph)
                    for ontology_id in relation_nodes
                ],
            }
        )
    return contexts


class EmbeddingResolver:
    def __init__(
        self,
        config: dict[str, Any],
        local_cache: Path,
        source_cache: Path | None = None,
    ):
        local_cache.parent.mkdir(parents=True, exist_ok=True)
        self.local = sqlite3.connect(local_cache)
        self.local.execute(
            "CREATE TABLE IF NOT EXISTS embeddings "
            "(cache_key TEXT PRIMARY KEY, vector BLOB NOT NULL, dimension INTEGER NOT NULL, "
            "source TEXT NOT NULL)"
        )
        self.local.commit()
        self.source = (
            sqlite3.connect(f"file:{source_cache}?mode=ro", uri=True)
            if source_cache is not None and source_cache.exists()
            else None
        )
        embedding_config = config["embedding"]
        self.api_key = embedding_config["api_key"]
        self.base_url = embedding_config.get("base_url") or None
        self.model = str(embedding_config.get("model") or EMBEDDING_MODEL)
        if self.model != EMBEDDING_MODEL:
            raise ValueError(f"Expected {EMBEDDING_MODEL}, got {self.model}")
        self.dimension = 3072
        self.client: OpenAI | None = None

    def close(self) -> None:
        self.local.close()
        if self.source is not None:
            self.source.close()
        if self.client is not None:
            self.client.close()

    def _key(self, text: str) -> str:
        return f"{self.model}\0{text}"

    def _get(self, connection: sqlite3.Connection, text: str) -> np.ndarray | None:
        row = connection.execute(
            "SELECT vector, dimension FROM embeddings WHERE cache_key = ?", (self._key(text),)
        ).fetchone()
        if row is None or int(row[1]) != self.dimension:
            return None
        vector = np.frombuffer(row[0], dtype=np.float32).copy()
        return vector if vector.shape == (self.dimension,) else None

    def resolve(self, texts: Sequence[str]) -> tuple[np.ndarray, dict[str, int]]:
        unique_texts = list(dict.fromkeys(str(text) for text in texts))
        vectors: dict[str, np.ndarray] = {}
        stats = {"local_cache": 0, "source_cache": 0, "api": 0, "api_batches": 0}
        missing: list[str] = []
        for text in unique_texts:
            vector = self._get(self.local, text)
            if vector is not None:
                vectors[text] = vector
                stats["local_cache"] += 1
                continue
            vector = self._get(self.source, text) if self.source is not None else None
            if vector is not None:
                vectors[text] = vector
                stats["source_cache"] += 1
                self.local.execute(
                    "INSERT OR REPLACE INTO embeddings(cache_key, vector, dimension, source) "
                    "VALUES (?, ?, ?, ?)",
                    (self._key(text), vector.tobytes(), self.dimension, "comparison_cache"),
                )
            else:
                missing.append(text)
        self.local.commit()
        if missing:
            if self.client is None:
                kwargs: dict[str, Any] = {"api_key": self.api_key, "max_retries": 3}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self.client = OpenAI(**kwargs)
            for start in range(0, len(missing), 128):
                batch = missing[start : start + 128]
                response = self.client.embeddings.create(model=self.model, input=batch)
                ordered = sorted(response.data, key=lambda item: item.index)
                generated = np.asarray(
                    [item.embedding for item in ordered], dtype=np.float32
                )
                if generated.shape != (len(batch), self.dimension):
                    raise ValueError(f"Unexpected embedding shape: {generated.shape}")
                for text, vector in zip(batch, generated):
                    vectors[text] = vector
                    self.local.execute(
                        "INSERT OR REPLACE INTO embeddings(cache_key, vector, dimension, source) "
                        "VALUES (?, ?, ?, ?)",
                        (self._key(text), vector.tobytes(), self.dimension, "api"),
                    )
                self.local.commit()
                stats["api"] += len(batch)
                stats["api_batches"] += 1
        matrix = np.vstack([vectors[str(text)] for text in texts]).astype(np.float32)
        return matrix, stats


def consolidate_retrieval(
    documents: Sequence[str],
    metadatas: Sequence[dict[str, Any] | None],
    distances: Sequence[float],
    top_k: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for document, metadata, distance in zip(documents, metadatas, distances):
        metadata = metadata or {}
        ontology_id = str(metadata.get("ontology_id") or metadata.get("hpo_id") or "")
        if not ontology_id or ontology_id in seen:
            continue
        seen.add(ontology_id)
        output.append(
            {
                "id": ontology_id,
                "score": round(1.0 - float(distance), 10),
                "matched_surface": str(document),
            }
        )
        if len(output) == top_k:
            break
    return output


def retrieve_many(
    collection: Any, vectors: np.ndarray, top_k: int = 20
) -> list[list[dict[str, Any]]]:
    requested = top_k * 4
    all_candidates: list[list[dict[str, Any]]] = []
    for start in range(0, len(vectors), RETRIEVAL_BATCH_SIZE):
        batch = vectors[start : start + RETRIEVAL_BATCH_SIZE]
        result = collection.query(
            query_embeddings=batch.tolist(),
            n_results=requested,
            include=["documents", "metadatas", "distances"],
        )
        for offset, (documents, metadatas, distances) in enumerate(
            zip(result["documents"], result["metadatas"], result["distances"])
        ):
            candidates = consolidate_retrieval(documents, metadatas, distances, top_k)
            if len(candidates) != top_k:
                query_vector = vectors[start + offset]
                base_requested = requested
                expanded = min(collection.count(), base_requested * 2)
                while len(candidates) < top_k and expanded > base_requested:
                    retry = collection.query(
                        query_embeddings=[query_vector.tolist()],
                        n_results=expanded,
                        include=["documents", "metadatas", "distances"],
                    )
                    candidates = consolidate_retrieval(
                        retry["documents"][0],
                        retry["metadatas"][0],
                        retry["distances"][0],
                        top_k,
                    )
                    if expanded >= collection.count():
                        break
                    expanded = min(collection.count(), expanded * 2)
                if len(candidates) != top_k:
                    raise RuntimeError(
                        f"Retrieved only {len(candidates)} unique concepts after expansion"
                    )
            all_candidates.append(candidates)
        print(
            f"retrieval {min(start + len(batch), len(vectors))}/{len(vectors)}",
            flush=True,
        )
    return all_candidates


def retrieval_records(
    samples: Sequence[dict[str, Any]],
    candidates: Sequence[Sequence[dict[str, Any]]],
    method: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sample, ranked in zip(samples, candidates):
        candidate_ids = [item["id"] for item in ranked]
        predicted = candidate_ids[0]
        records.append(
            {
                **sample,
                "method": method,
                "candidate_ids": candidate_ids,
                "candidate_scores": [item["score"] for item in ranked],
                "candidate_matched_surfaces": [
                    item["matched_surface"] for item in ranked
                ],
                "predicted_id": predicted,
                "no_match": False,
                "correct": predicted in set(sample["accepted_ids"]),
            }
        )
    return records


def candidate_payloads(
    candidate_ids: Sequence[str], terms: dict[str, OntologyTerm]
) -> list[dict[str, Any]]:
    return [
        {
            "id": ontology_id,
            "preferred_name": terms[ontology_id].name,
            "synonyms": list(terms[ontology_id].synonyms[:20]),
            "definition": terms[ontology_id].definition[:2000],
        }
        for ontology_id in candidate_ids
    ]


def build_lcr_task(
    record: dict[str, Any], terms: dict[str, OntologyTerm], stage: str, model: str
) -> dict[str, Any]:
    candidate_ids = list(record["candidate_ids"])
    input_payload = json.dumps(
        {
            "original_text": record["query"],
            "candidates": candidate_payloads(candidate_ids, terms),
        },
        ensure_ascii=False,
        indent=2,
    )
    user_prompt = USER_PROMPT_TEMPLATE.format(input_payload=input_payload)
    identity = {
        "stage": stage,
        "model": model,
        "temperature": LLM_TEMPERATURE,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "candidate_ids": candidate_ids,
    }
    return {
        "key": json.dumps(identity, ensure_ascii=False, sort_keys=True),
        "stage": stage,
        "model": model,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "allowed_ids": candidate_ids,
        "allow_no_match": True,
    }


def build_hgr_task(
    record: dict[str, Any],
    terms: dict[str, OntologyTerm],
    model: str,
    stage: str = "A4_HGR",
) -> dict[str, Any]:
    candidate_ids = list(record["ranking"])
    input_payload = json.dumps(
        {
            "original_text": record["query"],
            "candidates": candidate_payloads(candidate_ids, terms),
            "hpo_graph_context": record["hpo_graph_context"],
        },
        ensure_ascii=False,
        indent=2,
    )
    user_prompt = HPO_GRAPH_RERANK_PROMPT.format(input_payload=input_payload)
    identity = {
        "stage": stage,
        "model": model,
        "temperature": LLM_TEMPERATURE,
        "system_prompt": HPO_GRAPH_SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "candidate_ids": candidate_ids,
    }
    return {
        "key": json.dumps(identity, ensure_ascii=False, sort_keys=True),
        "stage": stage,
        "model": model,
        "system_prompt": HPO_GRAPH_SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "allowed_ids": candidate_ids,
        "allow_no_match": False,
    }


def parse_ranking(
    content: str, allowed_ids: Sequence[str], allow_no_match: bool
) -> tuple[list[str], bool]:
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        str(content or "").strip(),
        flags=re.IGNORECASE,
    )
    if cleaned in {"No Match", '"No Match"'}:
        if not allow_no_match:
            raise ValueError("HGR must return a complete ranking")
        return [], True
    payload = json.loads(cleaned)
    ranking = payload.get("ranking") if isinstance(payload, dict) else None
    if not isinstance(ranking, list):
        raise ValueError("LLM response has no ranking array")
    ranking = [str(value) for value in ranking]
    if len(ranking) != len(allowed_ids) or set(ranking) != set(allowed_ids):
        raise ValueError("LLM ranking is not a complete candidate permutation")
    return ranking, False


def repair_single_id_substitution(
    content: str, allowed_ids: Sequence[str]
) -> tuple[list[str], dict[str, str]] | None:
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        str(content or "").strip(),
        flags=re.IGNORECASE,
    )
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    ranking = payload.get("ranking") if isinstance(payload, dict) else None
    if not isinstance(ranking, list):
        return None
    ranking = [str(value) for value in ranking]
    allowed = set(allowed_ids)
    returned = set(ranking)
    missing = allowed - returned
    additional = returned - allowed
    if (
        len(ranking) != len(allowed_ids)
        or len(returned) != len(ranking)
        or len(missing) != 1
        or len(additional) != 1
    ):
        return None
    missing_id = next(iter(missing))
    additional_id = next(iter(additional))
    repaired = [missing_id if value == additional_id else value for value in ranking]
    return repaired, {
        "type": "single_id_substitution",
        "returned_id": additional_id,
        "replacement_id": missing_id,
    }


def llm_request_options(model: str) -> dict[str, Any]:
    if model in {"deepseek-v4-flash", "deepseek-v4-pro"}:
        return {
            "max_tokens": 2048,
            "extra_body": {"thinking": {"type": "disabled"}},
        }
    return {}


class LLMTaskRunner:
    def __init__(
        self,
        config: dict[str, Any],
        model: str,
        requests_per_minute: int = REQUESTS_PER_MINUTE,
    ):
        self.config = config
        self.model = model
        self.requests_per_minute = int(requests_per_minute)
        self.rate_limiter = GlobalRequestRateLimiter(self.requests_per_minute)
        self.thread_local = threading.local()

    def _client(self) -> OpenAI:
        if not hasattr(self.thread_local, "client"):
            kwargs: dict[str, Any] = {
                "api_key": self.config["api_key"],
                "max_retries": 0,
            }
            if self.config.get("base_url"):
                kwargs["base_url"] = self.config["base_url"]
            self.thread_local.client = OpenAI(**kwargs)
        return self.thread_local.client

    def request(self, task: dict[str, Any]) -> dict[str, Any]:
        last_error = ""
        last_raw_response = ""
        invalid_ranking_response = ""
        last_usage: Any = None
        attempt_errors: list[str] = []
        total_seconds = 0.0
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                self.rate_limiter.acquire()
                started = time.perf_counter()
                messages = [
                    {"role": "system", "content": task["system_prompt"]},
                    {"role": "user", "content": task["user_prompt"]},
                ]
                if invalid_ranking_response:
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": invalid_ranking_response,
                            },
                            {
                                "role": "user",
                                "content": (
                                    "The previous response was invalid because the ranking "
                                    "contained a duplicate, missing, or additional ID. Return "
                                    "a corrected JSON object containing every provided candidate "
                                    "ID exactly once. The only allowed IDs are: "
                                    + json.dumps(task["allowed_ids"], ensure_ascii=True)
                                ),
                            },
                        ]
                    )
                response = self._client().chat.completions.create(
                    model=self.model,
                    temperature=LLM_TEMPERATURE,
                    messages=messages,
                    **llm_request_options(self.model),
                )
                total_seconds += time.perf_counter() - started
                content = response.choices[0].message.content or ""
                last_raw_response = content
                last_usage = response.usage
                ranking, no_match = parse_ranking(
                    content, task["allowed_ids"], task["allow_no_match"]
                )
                usage = response.usage
                return {
                    "key": task["key"],
                    "stage": task["stage"],
                    "model": self.model,
                    "success": True,
                    "ranking": ranking,
                    "no_match": no_match,
                    "attempts": attempt,
                    "api_seconds": round(total_seconds, 6),
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(
                        getattr(usage, "completion_tokens", 0) or 0
                    ),
                    "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                    "raw_response": content,
                    "ranking_repair": {},
                    "error": "",
                    "attempt_errors": attempt_errors,
                    "completed_at": utc_now(),
                }
            except Exception as error:
                last_error = f"{type(error).__name__}: {error}"
                attempt_errors.append(last_error)
                if isinstance(error, ValueError) and last_raw_response:
                    invalid_ranking_response = last_raw_response
                if getattr(error, "status_code", None) == 429:
                    self.rate_limiter.pause(60.0)
                if attempt < MAX_ATTEMPTS:
                    time.sleep(2**attempt)
        repaired = repair_single_id_substitution(
            last_raw_response, task["allowed_ids"]
        )
        if repaired is not None:
            ranking, repair = repaired
            return {
                "key": task["key"],
                "stage": task["stage"],
                "model": self.model,
                "success": True,
                "ranking": ranking,
                "no_match": False,
                "attempts": MAX_ATTEMPTS,
                "api_seconds": round(total_seconds, 6),
                "prompt_tokens": int(
                    getattr(last_usage, "prompt_tokens", 0) or 0
                ),
                "completion_tokens": int(
                    getattr(last_usage, "completion_tokens", 0) or 0
                ),
                "total_tokens": int(getattr(last_usage, "total_tokens", 0) or 0),
                "raw_response": last_raw_response,
                "ranking_repair": repair,
                "error": "",
                "attempt_errors": attempt_errors,
                "completed_at": utc_now(),
            }
        return {
            "key": task["key"],
            "stage": task["stage"],
            "model": self.model,
            "success": False,
            "ranking": [],
            "no_match": False,
            "attempts": MAX_ATTEMPTS,
            "api_seconds": round(total_seconds, 6),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "raw_response": last_raw_response,
            "ranking_repair": {},
            "error": last_error,
            "attempt_errors": attempt_errors,
            "completed_at": utc_now(),
        }


def run_unique_tasks(
    tasks: Sequence[dict[str, Any]],
    cache_path: Path,
    runner: LLMTaskRunner,
    concurrency: int,
) -> dict[str, dict[str, Any]]:
    unique_tasks = {task["key"]: task for task in tasks}
    cached: dict[str, dict[str, Any]] = {}
    for item in load_jsonl(cache_path):
        if item.get("success"):
            cached[str(item["key"])] = item
    pending = [task for key, task in unique_tasks.items() if key not in cached]
    print(
        f"{cache_path.stem}: unique={len(unique_tasks)} cached={len(cached)} "
        f"pending={len(pending)} concurrency={concurrency}",
        flush=True,
    )
    failures: list[dict[str, Any]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_map = {executor.submit(runner.request, task): task for task in pending}
        for future in as_completed(future_map):
            result = future.result()
            append_jsonl(cache_path, result)
            completed += 1
            if result["success"]:
                cached[result["key"]] = result
            else:
                failures.append(result)
            if completed % 25 == 0 or completed == len(pending):
                print(
                    f"{cache_path.stem}: completed={completed}/{len(pending)} "
                    f"failures={len(failures)}",
                    flush=True,
                )
    if failures:
        raise RuntimeError(
            f"{len(failures)} LLM tasks failed; rerun the same command to retry"
        )
    missing = set(unique_tasks) - set(cached)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} successful LLM responses")
    return cached


def assemble_lcr_records(
    retrieval_records_input: Sequence[dict[str, Any]],
    tasks: Sequence[dict[str, Any]],
    responses: dict[str, dict[str, Any]],
    method: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for retrieval, task in zip(retrieval_records_input, tasks):
        response = responses[task["key"]]
        ranking = list(response["ranking"])
        predicted = ranking[0] if ranking else "No Match"
        records.append(
            {
                **retrieval,
                "method": method,
                "retrieval_method": retrieval["method"],
                "ranking": ranking,
                "no_match": bool(response["no_match"]),
                "predicted_id": predicted,
                "correct": predicted in set(retrieval["accepted_ids"]),
                "llm_response_key": task["key"],
                "llm_attempts": response["attempts"],
                "llm_api_seconds": response["api_seconds"],
                "prompt_tokens": response["prompt_tokens"],
                "completion_tokens": response["completion_tokens"],
                "total_tokens": response["total_tokens"],
                "raw_response": response["raw_response"],
                "llm_ranking_repair": response.get("ranking_repair", {}),
                "error": response["error"],
            }
        )
    return records


def method_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_dataset[record["dataset_key"]].append(record)
    per_dataset: dict[str, Any] = {}
    for dataset, values in sorted(by_dataset.items()):
        correct = sum(bool(value["correct"]) for value in values)
        per_dataset[dataset] = {
            "rows": len(values),
            "correct": correct,
            "accuracy": correct / len(values),
            "no_match": sum(bool(value.get("no_match")) for value in values),
        }
    macro = float(np.mean([value["accuracy"] for value in per_dataset.values()]))
    total_correct = sum(bool(record["correct"]) for record in records)
    return {
        "rows": len(records),
        "correct": total_correct,
        "micro_top1": total_correct / len(records),
        "macro_top1": macro,
        "no_match": sum(bool(record.get("no_match")) for record in records),
        "per_dataset": per_dataset,
    }


def paired_comparison(
    baseline: Sequence[dict[str, Any]],
    treatment: Sequence[dict[str, Any]],
    label: str,
) -> dict[str, Any]:
    left = {record["sample_id"]: record for record in baseline}
    right = {record["sample_id"]: record for record in treatment}
    if set(left) != set(right):
        raise ValueError(f"Paired sample mismatch for {label}")
    wrong_to_right = 0
    right_to_wrong = 0
    both_right = 0
    both_wrong = 0
    differences_by_dataset: dict[str, list[int]] = defaultdict(list)
    for sample_id in left:
        before = bool(left[sample_id]["correct"])
        after = bool(right[sample_id]["correct"])
        differences_by_dataset[left[sample_id]["dataset_key"]].append(
            int(after) - int(before)
        )
        if before and after:
            both_right += 1
        elif before and not after:
            right_to_wrong += 1
        elif not before and after:
            wrong_to_right += 1
        else:
            both_wrong += 1
    discordant = wrong_to_right + right_to_wrong
    p_value = (
        float(binomtest(wrong_to_right, discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrapped = np.zeros(BOOTSTRAP_ITERATIONS, dtype=np.float64)
    for differences in differences_by_dataset.values():
        values = np.asarray(differences, dtype=np.int8)
        indices = rng.integers(
            0, len(values), size=(BOOTSTRAP_ITERATIONS, len(values))
        )
        bootstrapped += values[indices].mean(axis=1)
    bootstrapped /= len(differences_by_dataset)
    baseline_macro = method_metrics(baseline)["macro_top1"]
    treatment_macro = method_metrics(treatment)["macro_top1"]
    return {
        "comparison": label,
        "macro_delta": treatment_macro - baseline_macro,
        "bootstrap_95_ci": [
            float(np.quantile(bootstrapped, 0.025)),
            float(np.quantile(bootstrapped, 0.975)),
        ],
        "mcnemar_exact_p": p_value,
        "wrong_to_right": wrong_to_right,
        "right_to_wrong": right_to_wrong,
        "both_right": both_right,
        "both_wrong": both_wrong,
        "discordant": discordant,
    }


def response_usage_for_keys(path: Path, keys: set[str]) -> dict[str, Any]:
    successful: dict[str, dict[str, Any]] = {}
    for item in load_jsonl(path):
        key = str(item.get("key", ""))
        if key in keys and item.get("success"):
            successful[key] = item
    missing = keys - set(successful)
    if missing:
        raise ValueError(f"{path.name} is missing {len(missing)} requested responses")
    return {
        "unique_successful_requests": len(successful),
        "attempts": sum(int(item.get("attempts", 0)) for item in successful.values()),
        "retries": sum(
            max(0, int(item.get("attempts", 0)) - 1) for item in successful.values()
        ),
        "api_seconds": round(
            sum(float(item.get("api_seconds", 0.0)) for item in successful.values()), 3
        ),
        "prompt_tokens": sum(
            int(item.get("prompt_tokens", 0)) for item in successful.values()
        ),
        "completion_tokens": sum(
            int(item.get("completion_tokens", 0)) for item in successful.values()
        ),
        "total_tokens": sum(
            int(item.get("total_tokens", 0)) for item in successful.values()
        ),
    }
