#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2016-2025.  SupportVectors AI Lab
#  -------------------------------------------------------------------------------------------------
"""FAISS index build and query for nearest-neighbor search (NumPy fallback if FAISS DLL fails)."""
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import faiss

    _FAISS_AVAILABLE = True
except (ImportError, OSError):
    faiss = None  # type: ignore[assignment, misc]
    _FAISS_AVAILABLE = False


class _NumpyFlatIPIndex:
    """Inner-product search for L2-normalized vectors (same semantics as faiss.IndexFlatIP)."""

    def __init__(self, vectors: np.ndarray) -> None:
        self._vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.ntotal = int(self._vectors.shape[0])

    def search(self, x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        sims = (self._vectors @ x.T).reshape(-1)
        k_eff = min(int(k), self.ntotal)
        if k_eff <= 0:
            return np.zeros((1, 0), dtype=np.float32), np.full((1, 0), -1, dtype=np.int64)
        part = np.argpartition(-sims, k_eff - 1)[:k_eff]
        order = part[np.argsort(-sims[part])]
        scores = sims[order].astype(np.float32)
        return scores.reshape(1, -1), order.astype(np.int64).reshape(1, -1)


def get_index_dir(name: str = "default") -> Path:
    base = Path(__file__).resolve().parent.parent.parent
    # Keep backward compatibility for older builds.
    if name == "default":
        index_dir = base / "index"
    else:
        index_dir = base / "index" / name
    index_dir.mkdir(parents=True, exist_ok=True)
    return index_dir


def build_index(
    embeddings: np.ndarray,
    ids: List[str],
    index_dir: Path | None = None,
    *,
    embedding_source: str | None = None,
    embedding_model_id: str | None = None,
) -> Path:
    """Build index from embeddings (N, D) float32; uses FAISS if available, else NumPy store."""
    index_dir = index_dir or get_index_dir()
    d = embeddings.shape[1]
    emb = embeddings.astype(np.float32)
    index_path = index_dir / "faiss_index.bin"
    vectors_path = index_dir / "vectors.npy"
    manifest_path = index_dir / "manifest.json"

    if _FAISS_AVAILABLE:
        assert faiss is not None
        index = faiss.IndexFlatIP(d)
        index.add(emb)
        faiss.write_index(index, str(index_path))
        if vectors_path.exists():
            vectors_path.unlink()
        manifest = {"ids": ids, "dim": d, "backend": "faiss"}
    else:
        if index_path.exists():
            index_path.unlink()
        np.save(vectors_path, emb)
        manifest = {"ids": ids, "dim": d, "backend": "numpy"}

    if embedding_source is not None:
        manifest["embedding_source"] = embedding_source
    if embedding_model_id is not None:
        manifest["embedding_model_id"] = embedding_model_id

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return index_dir


def load_index(index_dir: Path | None = None) -> Tuple[Any, List[str], Dict[str, Any]]:
    """Load index, id list, and manifest (FAISS index or NumPy-backed)."""
    index_dir = index_dir or get_index_dir()
    index_path = index_dir / "faiss_index.bin"
    vectors_path = index_dir / "vectors.npy"
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Index not found. Build it first (e.g. run scripts/build_index.py). "
            f"Expected: {manifest_path}"
        )
    with open(manifest_path) as f:
        manifest = json.load(f)
    id_list = manifest["ids"]
    backend = manifest.get("backend")
    if backend is None and index_path.exists():
        backend = "faiss"
    elif backend is None and vectors_path.exists():
        backend = "numpy"

    if backend == "numpy":
        if not vectors_path.exists():
            raise FileNotFoundError(f"NumPy index missing: {vectors_path}")
        vecs = np.load(vectors_path)
        return _NumpyFlatIPIndex(vecs), id_list, manifest

    if backend == "faiss" or backend is None:
        if not _FAISS_AVAILABLE:
            raise RuntimeError(
                "This index was built with FAISS, but FAISS failed to load (often a Windows DLL error). "
                "Install the latest Microsoft Visual C++ Redistributable (x64), then run: "
                "pip install --force-reinstall faiss-cpu. "
                "Or delete the index folder and run scripts/build_index.py to rebuild using the NumPy backend."
            )
        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. Re-run scripts/build_index.py."
            )
        assert faiss is not None
        return faiss.read_index(str(index_path)), id_list, manifest

    raise ValueError(f"Unknown index backend in manifest: {backend!r}")


def index_embedding_dim(index: Any) -> int:
    """Vector dimension D for a loaded FAISS index or NumPy-backed index."""
    if hasattr(index, "d"):
        return int(index.d)
    if hasattr(index, "_vectors"):
        return int(index._vectors.shape[1])
    raise TypeError(f"Unknown index type for dimension lookup: {type(index)!r}")


def query_index(
    index: Any,
    id_list: List[str],
    query_embedding: np.ndarray,
    k: int = 20,
) -> List[Tuple[str, float]]:
    """Returns list of (id, score) for top-k (score = cosine similarity for normalized vectors)."""
    query_embedding = query_embedding.astype(np.float32)
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    idx_d = index_embedding_dim(index)
    q_d = int(query_embedding.shape[1])
    if q_d != idx_d:
        raise RuntimeError(
            f"Embedding dimension mismatch: query has D={q_d} but the index expects D={idx_d}. "
            "Use 'Cluster multimodal' if you built the index with `python scripts/build_index.py` "
            "(default: cluster SigLIP). Use 'Local SigLIP' when the index was built with "
            "`python scripts/build_index.py --local` or otherwise uses the same model id and "
            "embedding dimension as `embeddings.encode_text` (see config `ray_cluster_api.text_embedding_model`)."
        )
    ntotal = int(getattr(index, "ntotal"))
    if ntotal == 0:
        return []
    scores, indices = index.search(query_embedding, min(k, ntotal))
    out = []
    for idx, sc in zip(indices[0], scores[0]):
        if idx >= 0 and idx < len(id_list):
            out.append((id_list[idx], float(sc)))
    return out
