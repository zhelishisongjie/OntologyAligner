from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


OAR_INPUT_DIMENSION = 3072
OAR_OUTPUT_DIMENSION = 3072
DEFAULT_OAR_MODEL_PATH = Path("models/oar_projection.pt")
DEFAULT_OAR_CHROMA_PATH = Path("chroma_db_hpo_a1_260623")
OAR_COLLECTION_NAME = "hpo_a1_3072_top1_20260623"
INDEX_BATCH_SIZE = 128


class OARLinearProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(
            OAR_INPUT_DIMENSION, OAR_OUTPUT_DIMENSION, bias=False
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(embeddings), p=2, dim=-1)


class OARRuntime:
    def __init__(self, model_path: Path, device: str | torch.device | None = None) -> None:
        self.model_path = model_path.resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"OAR weights are missing: {self.model_path}")
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = OARLinearProjection().to(self.device)
        state = torch.load(self.model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    @torch.inference_mode()
    def project(
        self, embeddings: np.ndarray | Sequence[Sequence[float]], batch_size: int = 256
    ) -> np.ndarray:
        values = np.asarray(embeddings, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != OAR_INPUT_DIMENSION:
            raise ValueError(
                f"OAR expects [N, {OAR_INPUT_DIMENSION}] embeddings, got {values.shape}"
            )
        parts: list[np.ndarray] = []
        for start in range(0, len(values), int(batch_size)):
            batch = torch.as_tensor(
                values[start : start + int(batch_size)],
                dtype=torch.float32,
                device=self.device,
            )
            parts.append(self.model(batch).cpu().numpy())
        return np.vstack(parts) if parts else np.empty((0, OAR_OUTPUT_DIMENSION), np.float32)


def load_default_oar_runtime(root: Path, device: str | None = None) -> OARRuntime:
    return OARRuntime(root / DEFAULT_OAR_MODEL_PATH, device=device)


def hpo_surfaces(terms: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    surfaces: list[tuple[str, str, str, str]] = []
    for term in terms.values():
        seen: set[str] = set()
        for text in (term.name, *term.synonyms):
            surface = str(text).strip()
            if surface and surface not in seen:
                surfaces.append((surface, term.ontology_id, term.name, term.definition))
                seen.add(surface)
    return surfaces


def ensure_oar_collection(
    root: Path,
    config: dict[str, Any],
    runtime: OARRuntime,
    rebuild: bool = False,
) -> Any:
    import chromadb
    import ontology_aligner_runtime as core

    surfaces = hpo_surfaces(core.load_ontology())
    client = chromadb.PersistentClient(path=str(root / DEFAULT_OAR_CHROMA_PATH))
    names = {item.name if hasattr(item, "name") else str(item) for item in client.list_collections()}
    if rebuild and OAR_COLLECTION_NAME in names:
        client.delete_collection(OAR_COLLECTION_NAME)
        names.remove(OAR_COLLECTION_NAME)
    if OAR_COLLECTION_NAME in names:
        collection = client.get_collection(OAR_COLLECTION_NAME)
        metadata = collection.metadata or {}
        if metadata.get("embedding_model") != core.EMBEDDING_MODEL:
            raise ValueError("Existing OAR index uses a different embedding model; rerun with --rebuild-index")
        if collection.count() > len(surfaces):
            raise ValueError("Existing OAR index has too many surfaces; rerun with --rebuild-index")
    else:
        collection = client.create_collection(
            OAR_COLLECTION_NAME,
            metadata={"embedding_model": core.EMBEDDING_MODEL, "build_status": "building"},
            configuration={"hnsw": {"space": "cosine"}},
        )
    start = collection.count()
    if start == len(surfaces):
        if (collection.metadata or {}).get("build_status") != "complete":
            collection.modify(metadata={"embedding_model": core.EMBEDDING_MODEL, "build_status": "complete"})
        return collection
    if start:
        last_id = f"surface_{start - 1:07d}"
        if not collection.get(ids=[last_id], include=[])["ids"]:
            raise ValueError("OAR index cannot be resumed; rerun with --rebuild-index")

    cache_path = root / ".cache" / "oar_surface_embeddings.sqlite3"
    resolver = core.EmbeddingResolver(config, cache_path, source_cache=None)
    try:
        for offset in range(start, len(surfaces), INDEX_BATCH_SIZE):
            batch = surfaces[offset : offset + INDEX_BATCH_SIZE]
            vectors, _ = resolver.resolve([item[0] for item in batch])
            projected = runtime.project(vectors)
            collection.add(
                ids=[f"surface_{offset + index:07d}" for index in range(len(batch))],
                documents=[item[0] for item in batch],
                metadatas=[
                    {"hpo_id": item[1], "preferred_label": item[2], "surface_index": offset + index}
                    for index, item in enumerate(batch)
                ],
                embeddings=projected.tolist(),
            )
            print(f"OAR index {offset + len(batch)}/{len(surfaces)}", flush=True)
    finally:
        resolver.close()
    collection.modify(metadata={"embedding_model": core.EMBEDDING_MODEL, "build_status": "complete"})
    return collection
