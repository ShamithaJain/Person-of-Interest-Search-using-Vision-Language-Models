# Person of Interest – What to do next

## 1. Install the new dependencies

From the project root (folder containing `pyproject.toml`):

```powershell
pip install -e .
```

This installs the added packages: Streamlit, PyTorch, FAISS, transformers, etc.

## 2. Add face images

- Create **`data/faces`** in the project root (if it doesn’t exist).
- Put face images there (`.jpg`, `.png`, `.webp`, or `.avif`), in the folder or in subfolders. The app indexes everything under `data/faces`.

## 3. Build the search index

From the project root:

```powershell
uv run python scripts/build_index.py
```

This loads all images in `data/faces`, encodes them with CLIP, and saves a FAISS index under **`index/`**.

## 4. Run the app

```powershell
uv run streamlit run src/person_of_intrest/app.py
```

Then:

- Describe a person in the text box (e.g. *“A woman with curly hair and a warm smile”*).
- Click **Search** to see the closest matches from your corpus.

## 5. Optional next steps (from the project spec)

- **DeepFace baseline**: Add a small script that computes face embeddings with DeepFace and compare retrieval vs CLIP.
- **SigLIP / Qwen-VL**: Swap or add SigLIP-2 for image–text embeddings and use Qwen-2.5-VL for narrative refinement.
- **Generation**: Use a vision–language model + image model to refine or generate an image from the narrative and top retrieval.
- **Logging**: Log queries and top results to `logs/` for failure analysis and improving prompts/models.
