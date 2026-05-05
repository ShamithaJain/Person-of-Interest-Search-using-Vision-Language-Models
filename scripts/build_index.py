#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2016-2025.  SupportVectors AI Lab
#  -------------------------------------------------------------------------------------------------
"""Build FAISS index from images in data/faces using cluster multimodal embeddings."""
import sys
from pathlib import Path

import numpy as np

# Add src to path when run as script
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "src"))

from person_of_intrest import data, index, sv_ray_cluster_api


def main() -> None:
    data_dir = data.get_data_dir()
    paths = data.list_image_paths(data_dir)
    if not paths:
        print("No images found in:", data_dir.resolve())
        if data_dir.exists():
            contents = list(data_dir.iterdir())
            if not contents:
                print("The folder is empty. Add .jpg, .png, or .avif images (or put them in a subfolder).")
            else:
                print("Contents (first level):", [p.name for p in contents[:20]])
                print("Supported extensions: .jpg .jpeg .png .bmp .webp .avif")
        else:
            print("The folder does not exist. It will be created when you add images.")
        sys.exit(1)

    ids = [str(p.relative_to(data_dir)) for p in paths]
    model_id = sv_ray_cluster_api._get_text_embedding_model()
    out_dir = index.get_index_dir("cluster")

    print("Preparing", len(paths), "image paths for cluster encoding...")
    image_inputs = [str(p.resolve()) for p in paths]
    print("Encoding images with cluster multimodal model...")
    vecs = sv_ray_cluster_api.embed_image(
        data_images=image_inputs,
        data_texts=[],
        model_name=model_id,
    )

    vecs = np.asarray(vecs, dtype=np.float32)
    print("Building FAISS index...")
    index.build_index(
        vecs,
        ids,
        index_dir=out_dir,
        embedding_source="cluster",
        embedding_model_id=model_id,
    )
    print(f"Done. Index saved under {out_dir}/")


if __name__ == "__main__":
    main()
