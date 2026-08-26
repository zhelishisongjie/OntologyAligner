from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


OAR_INPUT_DIMENSION = 3072
OAR_OUTPUT_DIMENSION = 3072
DEFAULT_OAR_RESULT_DIR = Path("results/a1_3072_260623")
DEFAULT_OAR_MODEL_PATH = DEFAULT_OAR_RESULT_DIR / "models/a1_best_top_1.pt"
DEFAULT_OAR_METADATA_PATH = DEFAULT_OAR_RESULT_DIR / "models/a1_best_top_1.json"
DEFAULT_OAR_CHROMA_PATH = Path("chroma_db_hpo_a1_260623")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class OARLinearProjection(nn.Module):
    """Bias-free 3072-to-3072 metric projection used by OAR."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(
            OAR_INPUT_DIMENSION, OAR_OUTPUT_DIMENSION, bias=False
        )
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(OAR_INPUT_DIMENSION))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(embeddings), p=2, dim=-1)


class OARRuntime:
    def __init__(
        self,
        model_path: Path,
        metadata_path: Path,
        device: str | torch.device | None = None,
    ) -> None:
        self.model_path = model_path.resolve()
        self.metadata_path = metadata_path.resolve()
        if not self.model_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError(
                "OAR model is missing; expected model and metadata under "
                f"{self.model_path.parent}"
            )
        self.metadata = json.loads(
            self.metadata_path.read_text(encoding="utf-8-sig")
        )
        self._validate_metadata()
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = OARLinearProjection().to(self.device)
        state = torch.load(self.model_path, map_location=self.device, weights_only=True)
        state.pop("initial_projection", None)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def _validate_metadata(self) -> None:
        if self.metadata.get("family") != "A1":
            raise ValueError("Configured model metadata is not compatible with OAR")
        group = self.metadata.get("group") or {}
        if int(group.get("output_dimension") or 0) != OAR_OUTPUT_DIMENSION:
            raise ValueError("OAR output dimension must be 3072")
        if self.metadata.get("identity", {}).get("initialization") != "identity":
            raise ValueError("OAR projection must use identity initialization")
        expected_hash = str(self.metadata.get("model_sha256") or "")
        actual_hash = file_sha256(self.model_path)
        if not expected_hash or actual_hash != expected_hash:
            raise ValueError("OAR model SHA-256 does not match its metadata")

    @property
    def model_sha256(self) -> str:
        return str(self.metadata["model_sha256"])

    @property
    def run_id(self) -> str:
        return str(self.metadata["run_id"])

    @property
    def catalog_fingerprint(self) -> str:
        return str(self.metadata["identity"]["catalog_fingerprint"])

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

    def collection_metadata(self) -> dict[str, Any]:
        return {
            "projection_family": "A1",
            "projection_run_id": self.run_id,
            "projection_model_sha256": self.model_sha256,
            "projection_input_dimension": OAR_INPUT_DIMENSION,
            "projection_output_dimension": OAR_OUTPUT_DIMENSION,
            "projection_bias": False,
            "projection_initialization": "identity",
            "catalog_fingerprint": self.catalog_fingerprint,
            "hnsw_space": "cosine",
            "distance_metric": "cosine",
        }


def load_default_oar_runtime(root: Path, device: str | None = None) -> OARRuntime:
    """Load the trained OAR projection from its legacy artifact location."""
    return OARRuntime(
        root / DEFAULT_OAR_MODEL_PATH,
        root / DEFAULT_OAR_METADATA_PATH,
        device=device,
    )
