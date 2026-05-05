import requests

base = 'http://10.0.10.51:8123/embed-text/v1/embeddings'
models = [
    'openai/clip-vit-base-patch32',
    'clip-vit-base-patch32',
    'clip-vit-large-patch14',
    'openai/clip-vit-large-patch14',
    'openai/clip-vit-base-patch16',
    'openai/clip-vit-base-patch32-multilingual-v1',
    'text-embedding-3-small',
    'text-embedding-3-large',
    'sentence-transformers/all-MiniLM-L6-v2',
]
for m in models:
    try:
        r = requests.post(base, json={'model': m, 'input': ['hello world']}, timeout=20)
        print(m, r.status_code, r.text[:500])
    except Exception as exc:
        print(m, 'ERROR', type(exc).__name__, exc)
