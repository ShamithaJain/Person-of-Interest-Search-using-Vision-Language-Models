#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2016-2025.  SupportVectors AI Lab
#  -------------------------------------------------------------------------------------------------
"""Text and image embeddings for multimodal retrieval (CLIP, SigLIP / SigLIP2 via Hugging Face)."""
import os
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_local_embedding_model_id() -> str:
    """Model id for local encoding.

    Priority:
    1) POI_LOCAL_EMBEDDING_MODEL
    2) config.yaml → ray_cluster_api.text_embedding_model
    3) Default aligned with typical cluster multimodal setup (SigLIP2).
    """
    env = os.getenv("POI_LOCAL_EMBEDDING_MODEL", "").strip()
    if env:
        return env

    config_path = Path(__file__).resolve().parents[2] / "config.yaml"
    if config_path.exists():
        try:
            import yaml

            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            mid = cfg.get("ray_cluster_api", {}).get("text_embedding_model")
            if isinstance(mid, str) and mid.strip():
                return mid.strip()
        except Exception:
            pass

    return "google/siglip2-base-patch16-224"


def _pooled_embedding(out: object) -> torch.Tensor:
    """Normalize HF outputs to a single (batch, dim) tensor."""
    if isinstance(out, torch.Tensor):
        return out
    po = getattr(out, "pooler_output", None)
    if po is not None:
        return po
    lhs = getattr(out, "last_hidden_state", None)
    if lhs is not None:
        return lhs[:, 0, :]
    raise TypeError(f"Unexpected model output type for embeddings: {type(out)!r}")


def _load_siglip2(model_id: str, local_files_only: bool) -> Tuple[torch.nn.Module, object]:
    """SigLIP2: bypass broken AutoTokenizer routing; use Siglip2Model for patch-sequence images."""
    from transformers import Siglip2Model, Siglip2Processor, Siglip2Tokenizer
    from transformers.models.siglip2.image_processing_siglip2 import Siglip2ImageProcessor

    tokenizer = Siglip2Tokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    image_processor = Siglip2ImageProcessor.from_pretrained(model_id, local_files_only=local_files_only)
    processor = Siglip2Processor(image_processor=image_processor, tokenizer=tokenizer)
    # Hugging Face may map this checkpoint to SiglipModel via AutoModel; Siglip2Model matches patch tokens.
    model = Siglip2Model.from_pretrained(
        model_id,
        local_files_only=local_files_only,
        ignore_mismatched_sizes=True,
    )
    model.to(_device())
    model.eval()
    return model, processor


def _load_auto_model(model_id: str) -> Tuple[torch.nn.Module, object]:
    """Load model + processor (CLIP, SigLIP, SigLIP2, …)."""
    mid = model_id.lower()

    if "siglip2" in mid:
        last_err: Exception | None = None
        for local_files_only in (True, False):
            try:
                return _load_siglip2(model_id, local_files_only=local_files_only)
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(
            f"Unable to load SigLIP2 model {model_id!r}. "
            "Ensure network access on first run, or download the model into the Hugging Face cache. "
            "If weights/config look corrupted, clear the HF cache folder for this repo and retry. "
            f"Original error: {last_err}"
        ) from last_err

    try:
        model = AutoModel.from_pretrained(model_id, local_files_only=True)
        processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    except Exception:
        try:
            model = AutoModel.from_pretrained(model_id)
            processor = AutoProcessor.from_pretrained(model_id)
        except Exception as e:
            raise RuntimeError(
                f"Unable to load embedding model {model_id!r}. "
                "Ensure network access on first run, or download the model into the Hugging Face cache. "
                f"Original error: {e}"
            ) from e

    model.to(_device())
    model.eval()
    return model, processor


_model_cache: Tuple[torch.nn.Module, object, str] | None = None


def get_embedder(model_id: str | None = None):
    """Return cached (model, processor) for the given or resolved model id."""
    global _model_cache
    resolved = model_id or _resolve_local_embedding_model_id()
    if _model_cache is None or _model_cache[2] != resolved:
        model, processor = _load_auto_model(resolved)
        _model_cache = (model, processor, resolved)
    return _model_cache[0], _model_cache[1]


def _tensor_text_features(model: torch.nn.Module, inputs: dict) -> torch.Tensor:
    with torch.no_grad():
        out = model.get_text_features(**inputs)
    return _pooled_embedding(out)


def _tensor_image_features(model: torch.nn.Module, inputs: dict) -> torch.Tensor:
    with torch.no_grad():
        out = model.get_image_features(**inputs)
    return _pooled_embedding(out)


# Converts text → vector
def encode_text(queries: Union[str, List[str]]) -> np.ndarray:
    """Encode text(s); returns (N, D) float32 normalized."""
    model, processor = get_embedder()
    if isinstance(queries, str):
        queries = [queries]
    inputs = processor(text=queries, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = _tensor_text_features(model, inputs)
    out = out / out.norm(dim=-1, keepdim=True)
    return out.cpu().float().numpy()


# Converts images → vector
def encode_images(images: np.ndarray) -> np.ndarray:
    """Encode images (N, H, W, 3) uint8; returns (N, D) float32 normalized."""
    model, processor = get_embedder()
    pil_list = [Image.fromarray(x) for x in images]
    inputs = processor(images=pil_list, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    features = _tensor_image_features(model, inputs)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().float().numpy()
