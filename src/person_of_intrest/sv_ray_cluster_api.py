#  ------------------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------------------

import base64
import os
from functools import wraps
from pathlib import Path
from urllib.parse import urlparse, unquote
import requests


# base_url is the base URL of the SupportVectors Ray cluster API which happens to be 10.0.10.51:8123 today.
def _get_base_url() -> str:
    """Resolve Ray cluster API base URL.

    Priority:
    1) environment variable SV_RAY_CLUSTER_BASE_URL
    2) config.yaml via person_of_intrest.config (ray_cluster_api.base_url)
    """
    env = os.getenv("SV_RAY_CLUSTER_BASE_URL")
    if env:
        return env.rstrip("/")
    try:
        from person_of_intrest import config  # lazy import; may not exist for all users

        return str(config["ray_cluster_api"]["base_url"]).rstrip("/")
    except Exception as e:
        config_path = Path(__file__).resolve().parents[2] / "config.yaml"
        if config_path.exists():
            try:
                import yaml

                cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                base_url = cfg.get("ray_cluster_api", {}).get("base_url")
                if base_url:
                    return str(base_url).rstrip("/")
            except Exception:
                pass

        raise RuntimeError(
            "Ray cluster base URL not configured. Set environment variable "
            "SV_RAY_CLUSTER_BASE_URL (e.g. http://10.0.10.51:8123) or add "
            "ray_cluster_api.base_url to config.yaml. "
            f"Looked for config at: {config_path}"
        ) from e


def _get_text_embedding_model(model_name: str | None = None) -> str:
    """Resolve the cluster text embedding model from environment, config, or default."""
    if model_name:
        return model_name

    env_model = os.getenv("SV_RAY_CLUSTER_TEXT_EMBEDDING_MODEL")
    if env_model:
        return env_model

    try:
        from person_of_intrest import config  # lazy import
        return str(config["ray_cluster_api"]["text_embedding_model"])
    except Exception:
        return "openai/clip-vit-base-patch32"


def _embed_text_url() -> str:
    base_url = _get_base_url()
    return f"{base_url}/embed-text/v1/embeddings"


def _embed_image_url() -> str:
    base_url = _get_base_url()
    return f"{base_url}/embed-image/v1/embeddings"


def _chat_base_url() -> str:
    base_url = _get_base_url()
    return f"{base_url}/v1"

def _is_http_url(image_source: str) -> bool:
    parsed = urlparse(image_source)
    return parsed.scheme in {"http", "https"}

def _convert_image_to_base64(image_source: str) -> str:
    """Convert an image from URL or local path to base64.

    Args:
        image_source (str): URL (http/https) or local file path.

    Returns:
        str: Base64-encoded image bytes (no data URI prefix).
    """
    if _is_http_url(image_source):
        response = requests.get(image_source, timeout=30)
        response.raise_for_status()
        image_bytes = response.content
    else:
        parsed = urlparse(image_source)
        if parsed.scheme == "file":
            file_path = unquote(parsed.path)
        else:
            file_path = image_source
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image not found at path: {file_path}")
        with open(file_path, "rb") as image_file:
            image_bytes = image_file.read()

    return base64.b64encode(image_bytes).decode("utf-8")
def embed_text(
    data_sentences: list[str],
    model_name: str | None = None,
    batch_size: int = 50
) -> list[list[float]]:
    """
    Embed text using the SupportVectors Ray cluster API.

    Routing:
    - Multimodal models (CLIP, SigLIP) → /embed-image (input_type="auto")
    - Text-only models → /embed-text

    Args:
        data_sentences: list of input strings
        model_name: embedding model name (resolved from config/env if None)
        batch_size: request batch size

    Returns:
        List of embedding vectors
    """

    embeddings: list[list[float]] = []
    model_name = _get_text_embedding_model(model_name)

    # Detect multimodal models (CLIP, SigLIP, future-safe)
    model_lower = model_name.lower()
    is_multimodal = any(k in model_lower for k in ["clip", "siglip"])

    endpoint = _embed_image_url() if is_multimodal else _embed_text_url()

    for start in range(0, len(data_sentences), batch_size):
        batch = data_sentences[start:start + batch_size]

        payload = (
            {
                "model": model_name,
                "input_type": "auto",   # ✅ required for multimodal
                "input": batch
            }
            if is_multimodal
            else {
                "model": model_name,
                "input": batch
            }
        )

        response = requests.post(endpoint, json=payload, timeout=60)

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"[embed_text] Request failed | model={model_name} | "
                f"endpoint={endpoint} | status={response.status_code} | "
                f"body={response.text!r}"
            ) from exc

        data = response.json().get("data")
        if data is None:
            raise RuntimeError(
                f"[embed_text] Missing 'data' in response | body={response.text!r}"
            )

        embeddings.extend(item["embedding"] for item in data)
        print(f"Processed batch {start}–{start + len(batch)}")

    return embeddings
    
def embed_image(data_images: list[str], data_texts: list[str], model_name: str, batch_size: int = 50) -> list[list[float]]:
    """Embed image or text using the multi-modal models supported by the SupportVectors Ray cluster API

    Args:
        data_images (List[str]): List of image_urls to embed
        data_texts (List[str]): List of text to embed
        model_name (str): Name of the multi-modal model (could be a CLIP or SigLip) to use
        batch_size (int): Batch size to use for sending requests to the backend API. Default is 50.
    Returns:
        List[List[float]]: List of embeddings
    """
    # Convert the list of image_urls into a list of base64 encoded images
    base64_images = [_convert_image_to_base64(image_url) for image_url in data_images]
    # Concatenate this list with the list of text
    data_inputs = base64_images + data_texts
    # Send the request to the backend API in batches
    embeddings = []
    for start in range(0, len(data_inputs), batch_size):
        batch = data_inputs[start:start + batch_size]
        response = requests.post(
            _embed_image_url(),
            json={"model": model_name, "input_type": "auto", "input": batch},
            timeout=120,
        )
        response.raise_for_status()
        embedding_response = response.json()["data"]
        print(f"Batch {start} to {start + batch_size} processed successfully")
        embeddings.extend([item["embedding"] for item in embedding_response])

    return embeddings

def get_sv_openai_client():
    """Create an OpenAI client for the LLM served on the SupportVectors Ray cluster.
    Currently, the only supported model is "openai/gpt-oss-20b".  So, only pass this model
    when using this client for chat completions.
    Returns:
        OpenAI: An OpenAI client for the LLM (openai/gpt-oss-20b) served on the SupportVectors Ray cluster.
    """
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("Missing dependency 'openai'. Install it to use chat features.") from e
    return OpenAI(base_url=_chat_base_url(), api_key="sv-openai-api-key")

def sv_openai_completion(**kwargs):
    """
    Use the SV OpenAI completion function to get a response from the LLM served on the SupportVectors Ray cluster.
    Args:
        **kwargs: Arbitrary keyword arguments
    Returns:
        dict: The response from the LLM
    """
    client = get_sv_openai_client()
    kwargs.pop("model", None)
    return client.chat.completions.create(model="openai/gpt-oss-20b", **kwargs)

# define an sv_completion function that uses the litellm completion but hard-codes the model, api_base, and api_key
# and leaves the rest for the caller to fill in.
def sv_completion(**kwargs):
    """ 
    Use the litellm completion function to get a response from the LLM served on the SupportVectors Ray cluster.
    Args:
        **kwargs: Arbitrary keyword arguments
    Returns:
        dict: The response from the LLM
    """
    try:
        from litellm import completion  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency 'litellm'. Install it to use chat features.") from e

    kwargs.pop("model", None)
    kwargs.pop("api_base", None)
    kwargs.pop("api_key", None)
    return completion(
        model="openai/" + "openai/gpt-oss-20b",
        api_base=_chat_base_url(),
        api_key="sv-openai-api-key",
        **kwargs,
    )

# Below are some test functions to test the embed_text, embed_image, openai chat, and litellm chat functions.
def test_embed_text(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 50):
    """Test the embed_text function"""
    data_sentences = ["Hello, world!", "This is a test sentence.", "This is another test sentence."]
    embeddings = embed_text(data_sentences, model_name, batch_size)
    print(f"Generated {len(embeddings)} embeddings for the input sentences")
    print(f"Size of the embeddings: {len(embeddings[0])}")
    print(f"First 10 values of the first embedding: {embeddings[0][:10]}")

def test_embed_image(model_name: str = "google/siglip2-base-patch16-224", batch_size: int = 50):
    """Test the embed_image function"""
    data_images = ["https://supportvectors.ai/logo-poster-transparent.png"]
    data_texts = ["SupportVectors offers courses on AI/ML"]
    embeddings = embed_image(data_images, data_texts, model_name, batch_size)
    print(f"Generated {len(embeddings)} embeddings for the input images and text")
    print(f"Size of the embeddings: {len(embeddings[0])}")
    print(f"First 10 values of the first embedding: {embeddings[0][:10]}")

def test_openai_chat():
    """Test the openai chat function"""
    response = sv_openai_completion(messages=[{"role": "user", "content": "Why is the sky blue?"}])
    print(response.choices[0].message.content)

def test_litellm_chat():
    """Test the litellm chat function"""
    response = sv_completion(messages=[{"role": "user", "content": "Tell me a new joke"}])
    print(response.choices[0].message.content)
    
if __name__ == "__main__":
    print("Testing embed_text function")
    test_embed_text()
    print("--------------------------------")
    print("Testing embed_image function")
    test_embed_image()
    print("--------------------------------")
    print("Testing openai chat function")
    test_openai_chat()
    print("--------------------------------")
    print("Testing litellm chat function")
    test_litellm_chat()
    print("--------------------------------")