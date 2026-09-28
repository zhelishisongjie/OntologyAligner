from __future__ import annotations

import json
import math
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import chromadb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ontology_aligner_runtime as core

from . import config, runtime


class OARProjection(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = int(dimension)
        self.projection = nn.Linear(self.dimension, self.dimension, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(self.dimension))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(values), p=2, dim=-1)


def normalize_surface(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().casefold())


def model_dir(spec: config.BackboneSpec) -> Path:
    return config.RESULTS_DIR / "oar_models" / spec.key


def model_path(spec: config.BackboneSpec) -> Path:
    return model_dir(spec) / "oar_model.pt"


def metadata_path(spec: config.BackboneSpec) -> Path:
    return model_dir(spec) / "oar_metadata.json"


def mining_path(spec: config.BackboneSpec) -> Path:
    return config.TRAINING_CACHE_DIR / spec.key / "training_lookups.npz"


def mining_metadata_path(spec: config.BackboneSpec) -> Path:
    return config.TRAINING_CACHE_DIR / spec.key / "mining_metadata.json"


def projected_collection_name(spec: config.BackboneSpec) -> str:
    return f"hpo_oar_ablation_{spec.key}_20260623"


def set_determinism(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def ancestor_sets(graph: core.HPOGraphIndex) -> dict[str, frozenset[str]]:
    cache: dict[str, frozenset[str]] = {}

    def visit(ontology_id: str, active: set[str]) -> frozenset[str]:
        if ontology_id in cache:
            return cache[ontology_id]
        if ontology_id in active:
            return frozenset()
        active.add(ontology_id)
        values: set[str] = set()
        for parent in graph.parents.get(ontology_id, ()):
            values.add(parent)
            values.update(visit(parent, active))
        active.remove(ontology_id)
        cache[ontology_id] = frozenset(values)
        return cache[ontology_id]

    nodes = set(graph.names) | set(graph.parents)
    for node in nodes:
        visit(node, set())
    return cache


def surface_groups(
    metadatas: Sequence[dict[str, Any]], documents: Sequence[str]
) -> tuple[list[str], dict[str, list[int]], np.ndarray, list[int]]:
    concepts = [
        str(metadata.get("hpo_id") or metadata.get("ontology_id") or "")
        for metadata in metadatas
    ]
    if any(not value for value in concepts):
        raise ValueError("Surface collection contains an empty HPO ID")
    by_concept: dict[str, list[int]] = defaultdict(list)
    for index, concept in enumerate(concepts):
        by_concept[concept].append(index)
    anchors = np.asarray(
        [
            index
            for indexes in by_concept.values()
            if len(indexes) > 1
            for index in indexes
        ],
        dtype=np.int32,
    )
    representatives: dict[tuple[str, str], int] = {}
    for index in anchors:
        key = (concepts[int(index)], normalize_surface(documents[int(index)]))
        representatives.setdefault(key, int(index))
    return concepts, by_concept, anchors, list(representatives.values())


def mine_hard_negatives(
    spec: config.BackboneSpec,
    collection: Any,
    surface_matrix: np.ndarray,
    documents: Sequence[str],
    metadatas: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    cache_path = mining_path(spec)
    metadata_file = mining_metadata_path(spec)
    if cache_path.exists() and metadata_file.exists():
        arrays = np.load(cache_path)
        metadata = json.loads(metadata_file.read_text(encoding="utf-8-sig"))
        return (
            arrays["anchors"],
            arrays["positive_lookup"],
            arrays["negative_lookup"],
            metadata,
        )

    concepts, by_concept, anchors, representatives = surface_groups(
        metadatas, documents
    )
    graph = core.load_hpo_graph()
    ancestors = ancestor_sets(graph)
    max_positive = max(len(indexes) - 1 for indexes in by_concept.values())
    max_surfaces = max(len(indexes) for indexes in by_concept.values())
    positive_lookup = np.full((len(anchors), max_positive), -1, dtype=np.int32)
    negative_lookup = np.full(
        (len(anchors), config.OAR_NEGATIVE_CONCEPTS, max_surfaces),
        -1,
        dtype=np.int32,
    )
    anchor_positions = {int(anchor): index for index, anchor in enumerate(anchors)}
    duplicate_anchor_positions: dict[tuple[str, str], list[int]] = defaultdict(list)
    for position, anchor in enumerate(anchors):
        key = (concepts[int(anchor)], normalize_surface(documents[int(anchor)]))
        duplicate_anchor_positions[key].append(position)
        positives = [
            index for index in by_concept[concepts[int(anchor)]] if index != int(anchor)
        ]
        positive_lookup[position, : len(positives)] = positives

    started = time.perf_counter()
    selected_counts: list[int] = []
    filter_counts: dict[str, int] = defaultdict(int)
    for start in range(0, len(representatives), 128):
        batch_indices = representatives[start : start + 128]
        result = collection.query(
            query_embeddings=surface_matrix[batch_indices].tolist(),
            n_results=config.OAR_HARD_NEGATIVE_QUERY_K,
            include=["documents", "metadatas", "distances"],
        )
        for query_offset, anchor in enumerate(batch_indices):
            anchor_concept = concepts[anchor]
            anchor_normalized = normalize_surface(documents[anchor])
            selected: list[str] = []
            seen: set[str] = set()
            for document, metadata in zip(
                result["documents"][query_offset], result["metadatas"][query_offset]
            ):
                metadata = metadata or {}
                candidate = str(
                    metadata.get("hpo_id") or metadata.get("ontology_id") or ""
                )
                if candidate == anchor_concept:
                    filter_counts["anchor_concept"] += 1
                    continue
                if normalize_surface(str(document)) == anchor_normalized:
                    filter_counts["normalized_cross_id_text"] += 1
                    continue
                if (
                    candidate in ancestors.get(anchor_concept, frozenset())
                    or anchor_concept in ancestors.get(candidate, frozenset())
                ):
                    filter_counts["ancestor_or_descendant"] += 1
                    continue
                if candidate in seen:
                    filter_counts["duplicate_concept"] += 1
                    continue
                seen.add(candidate)
                if len(selected) < config.OAR_NEGATIVE_CONCEPTS:
                    selected.append(candidate)
                    filter_counts["selected"] += 1
                else:
                    filter_counts["beyond_negative_limit"] += 1
            key = (anchor_concept, anchor_normalized)
            for anchor_position in duplicate_anchor_positions[key]:
                for concept_slot, concept in enumerate(selected):
                    indexes = by_concept[concept]
                    negative_lookup[
                        anchor_position, concept_slot, : len(indexes)
                    ] = indexes
            selected_counts.extend([len(selected)] * len(duplicate_anchor_positions[key]))
        completed = min(start + len(batch_indices), len(representatives))
        if completed % 1024 < len(batch_indices) or completed == len(representatives):
            print(f"{spec.key}: hard negatives {completed}/{len(representatives)}", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        anchors=anchors,
        positive_lookup=positive_lookup,
        negative_lookup=negative_lookup,
    )
    metadata = {
        "backbone": spec.key,
        "embedding_model": spec.model_name,
        "anchor_surface_instance_count": int(len(anchors)),
        "unique_anchor_count": int(len(representatives)),
        "positive_lookup_shape": list(positive_lookup.shape),
        "negative_lookup_shape": list(negative_lookup.shape),
        "hard_negative_query_k": config.OAR_HARD_NEGATIVE_QUERY_K,
        "negative_concept_limit": config.OAR_NEGATIVE_CONCEPTS,
        "filter_rule": (
            "exclude anchor concept, normalized cross-ID text, and any ancestor or "
            "descendant; retain first surface per concept"
        ),
        "filter_reason_counts": dict(filter_counts),
        "negative_count_distribution": {
            "minimum": int(min(selected_counts)),
            "maximum": int(max(selected_counts)),
            "mean": float(np.mean(selected_counts)),
            "median": float(np.median(selected_counts)),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "created_at": core.utc_now(),
    }
    core.write_json(metadata_file, metadata)
    return anchors, positive_lookup, negative_lookup, metadata


def batch_scores(
    model: OARProjection,
    surface_matrix: np.ndarray,
    anchor_indexes: np.ndarray,
    positive_lookup: np.ndarray,
    negative_lookup: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    batch_size = len(anchor_indexes)
    pair_anchor: list[np.ndarray] = []
    pair_slot: list[np.ndarray] = []
    pair_surface: list[np.ndarray] = []
    for batch_index in range(batch_size):
        positives = positive_lookup[batch_index]
        positives = positives[positives >= 0]
        pair_anchor.append(np.full(len(positives), batch_index, dtype=np.int64))
        pair_slot.append(np.zeros(len(positives), dtype=np.int64))
        pair_surface.append(positives.astype(np.int64, copy=False))
        for negative_slot in range(config.OAR_NEGATIVE_CONCEPTS):
            negatives = negative_lookup[batch_index, negative_slot]
            negatives = negatives[negatives >= 0]
            if not len(negatives):
                continue
            pair_anchor.append(
                np.full(len(negatives), batch_index, dtype=np.int64)
            )
            pair_slot.append(
                np.full(len(negatives), negative_slot + 1, dtype=np.int64)
            )
            pair_surface.append(negatives.astype(np.int64, copy=False))
    flat_surfaces = np.concatenate(pair_surface)
    unique_surfaces, inverse = np.unique(flat_surfaces, return_inverse=True)
    anchor_tensor = torch.from_numpy(surface_matrix[anchor_indexes]).to(device)
    candidate_tensor = torch.from_numpy(surface_matrix[unique_surfaces]).to(device)
    projected_anchors = model(anchor_tensor)
    projected_candidates = model(candidate_tensor)
    anchor_pairs = torch.from_numpy(np.concatenate(pair_anchor)).to(device)
    slot_pairs = torch.from_numpy(np.concatenate(pair_slot)).to(device)
    inverse_tensor = torch.from_numpy(inverse).to(device)
    similarities = (
        projected_anchors[anchor_pairs] * projected_candidates[inverse_tensor]
    ).sum(dim=1)
    flat_indexes = anchor_pairs * (config.OAR_NEGATIVE_CONCEPTS + 1) + slot_pairs
    scores = torch.full(
        (batch_size * (config.OAR_NEGATIVE_CONCEPTS + 1),),
        -torch.inf,
        dtype=similarities.dtype,
        device=device,
    )
    scores.scatter_reduce_(0, flat_indexes, similarities, reduce="amax", include_self=True)
    return scores.view(batch_size, config.OAR_NEGATIVE_CONCEPTS + 1)


def train_oar(spec: config.BackboneSpec) -> tuple[Path, dict[str, Any]]:
    if not spec.train_oar:
        raise ValueError(f"{spec.key} uses the formal OAR and must not be retrained")
    if model_path(spec).exists() and metadata_path(spec).exists():
        metadata = json.loads(metadata_path(spec).read_text(encoding="utf-8-sig"))
        if (
            metadata.get("input_dimension") == spec.dimension
            and metadata.get("checkpoint_epoch") == config.OAR_CHECKPOINT_EPOCH
            and metadata.get("seed") == config.SEED
        ):
            return model_path(spec), metadata

    set_determinism()
    collection = runtime.raw_collection(spec)
    surface_matrix, documents, metadatas, _ = runtime.load_collection_matrix(
        collection, spec.dimension
    )
    anchors, positives, negatives, mining_metadata = mine_hard_negatives(
        spec, collection, surface_matrix, documents, metadatas
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OARProjection(spec.dimension).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.OAR_LEARNING_RATE,
        weight_decay=config.OAR_WEIGHT_DECAY,
    )
    steps_per_epoch = math.ceil(len(anchors) / config.OAR_BATCH_SIZE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps_per_epoch * config.OAR_EPOCHS
    )
    identity = torch.eye(spec.dimension, device=device)
    generator = np.random.default_rng(config.SEED)
    checkpoint_state: dict[str, torch.Tensor] | None = None
    epoch_losses: list[float] = []
    started = time.perf_counter()
    for epoch in range(1, config.OAR_EPOCHS + 1):
        order = generator.permutation(len(anchors))
        running_loss = 0.0
        seen = 0
        for start in range(0, len(order), config.OAR_BATCH_SIZE):
            positions = order[start : start + config.OAR_BATCH_SIZE]
            batch_anchors = anchors[positions]
            scores = batch_scores(
                model,
                surface_matrix,
                batch_anchors,
                positives[positions],
                negatives[positions],
                device,
            )
            targets = torch.zeros(len(positions), dtype=torch.long, device=device)
            loss = F.cross_entropy(scores / config.OAR_TEMPERATURE, targets)
            regularization = (model.projection.weight - identity).square().mean()
            loss = loss + config.OAR_IDENTITY_REGULARIZATION * regularization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.OAR_GRADIENT_CLIP)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.detach()) * len(positions)
            seen += len(positions)
            step = start // config.OAR_BATCH_SIZE + 1
            if step % 100 == 0 or step == steps_per_epoch:
                print(
                    f"{spec.key}: epoch={epoch}/{config.OAR_EPOCHS} "
                    f"step={step}/{steps_per_epoch} loss={running_loss / seen:.6f}",
                    flush=True,
                )
        epoch_losses.append(running_loss / seen)
        checkpoint = config.TRAINING_CACHE_DIR / spec.key / f"epoch_{epoch}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "dimension": spec.dimension,
                "loss": epoch_losses[-1],
            },
            checkpoint,
        )
        if epoch == config.OAR_CHECKPOINT_EPOCH:
            checkpoint_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    if checkpoint_state is None:
        raise RuntimeError("Epoch 5 checkpoint was not captured")

    model_dir(spec).mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": checkpoint_state,
            "input_dimension": spec.dimension,
            "output_dimension": spec.dimension,
            "bias": False,
        },
        model_path(spec),
    )
    metadata = {
        "backbone_key": spec.key,
        "embedding_model": spec.model_name,
        "revision": spec.revision,
        "pooling": spec.pooling,
        "input_dimension": spec.dimension,
        "output_dimension": spec.dimension,
        "bias": False,
        "initialization": "identity",
        "training_objective": "concept-level multi-positive InfoNCE",
        "concept_aggregation": "max cosine similarity",
        "optimizer": "AdamW",
        "scheduler": "cosine",
        "learning_rate": config.OAR_LEARNING_RATE,
        "temperature": config.OAR_TEMPERATURE,
        "weight_decay": config.OAR_WEIGHT_DECAY,
        "identity_regularization": config.OAR_IDENTITY_REGULARIZATION,
        "gradient_clip": config.OAR_GRADIENT_CLIP,
        "batch_size": config.OAR_BATCH_SIZE,
        "epochs_trained": config.OAR_EPOCHS,
        "checkpoint_epoch": config.OAR_CHECKPOINT_EPOCH,
        "seed": config.SEED,
        "dtype": "float32",
        "epoch_losses": epoch_losses,
        "mining": mining_metadata,
        "training_seconds": time.perf_counter() - started,
        "raw_collection": spec.collection_name,
        "raw_collection_metadata": collection.metadata,
        "created_at": core.utc_now(),
    }
    core.write_json(metadata_path(spec), metadata)
    del model, surface_matrix
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model_path(spec), metadata


def load_projection(
    spec: config.BackboneSpec, device: str | None = None
) -> tuple[OARProjection, dict[str, Any], torch.device]:
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    metadata = json.loads(metadata_path(spec).read_text(encoding="utf-8-sig"))
    payload = torch.load(model_path(spec), map_location="cpu", weights_only=False)
    model = OARProjection(spec.dimension)
    model.load_state_dict(payload["state_dict"])
    model.to(target).eval()
    return model, metadata, target


@torch.inference_mode()
def project_matrix(
    model: OARProjection,
    matrix: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    output = np.empty_like(matrix, dtype=np.float32)
    for start in range(0, len(matrix), batch_size):
        values = torch.from_numpy(matrix[start : start + batch_size]).to(device)
        output[start : start + len(values)] = model(values).cpu().numpy()
    return output


def build_projected_collection(spec: config.BackboneSpec) -> tuple[Any, dict[str, Any]]:
    _, model_metadata = train_oar(spec)
    client = chromadb.PersistentClient(path=str(config.PROJECTED_CHROMA_PATH))
    name = projected_collection_name(spec)
    names = {item.name for item in client.list_collections()}
    if name in names:
        existing = client.get_collection(name)
        metadata = existing.metadata or {}
        if existing.count() == 44_814:
            return existing, dict(metadata)
        raise ValueError(f"Projected collection identity conflict: {name}")

    raw = runtime.raw_collection(spec)
    surface_matrix, documents, metadatas, ids = runtime.load_collection_matrix(
        raw, spec.dimension
    )
    model, _, device = load_projection(spec)
    metadata = {
        "ontology": "hpo",
        "ontology_release": "2026-06-23",
        "embedding_model": spec.model_name,
        "model_revision": spec.revision,
        "pooling": spec.pooling,
        "source_dimension": spec.dimension,
        "projection_input_dimension": spec.dimension,
        "projection_output_dimension": spec.dimension,
        "projection_bias": False,
        "projection_initialization": "identity",
        "projection_checkpoint_epoch": config.OAR_CHECKPOINT_EPOCH,
        "projection_seed": config.SEED,
        "expected_surface_count": 44_814,
        "distance_metric": "cosine",
        "hnsw_space": "cosine",
        "build_status": "building",
    }
    collection = client.create_collection(
        name, metadata=metadata, configuration={"hnsw": {"space": "cosine"}}
    )
    for start in range(0, len(surface_matrix), 256):
        stop = min(start + 256, len(surface_matrix))
        projected = project_matrix(
            model, surface_matrix[start:stop], device, batch_size=256
        )
        collection.add(
            ids=ids[start:stop],
            documents=documents[start:stop],
            metadatas=metadatas[start:stop],
            embeddings=projected.tolist(),
        )
        if stop % 4096 < 256 or stop == len(surface_matrix):
            print(f"{spec.key}: projected index {stop}/{len(surface_matrix)}", flush=True)
    metadata["build_status"] = "complete"
    metadata["final_surface_count"] = collection.count()
    collection.modify(metadata=metadata)
    del model, surface_matrix
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return collection, metadata


def project_queries(spec: config.BackboneSpec, vectors: np.ndarray) -> np.ndarray:
    model, _, device = load_projection(spec)
    output = project_matrix(model, vectors, device)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output
