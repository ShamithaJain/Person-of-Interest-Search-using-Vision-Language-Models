#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2016-2025.  SupportVectors AI Lab
#  -------------------------------------------------------------------------------------------------
"""Streamlit UI: describe a person, search face corpus, and view top matches."""
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
src_path = root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import numpy as np
import streamlit as st

from person_of_intrest import data, index
from person_of_intrest.rerank import infer_constraints
from person_of_intrest import sv_ray_cluster_api



def _ensure_local_package_path() -> None:
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def _resolve_image_path(data_dir: Path, img_id: str) -> Path:
    return (data_dir / img_id).resolve()


def _display_results(data_dir: Path, results: list[tuple[str, float]]) -> None:
    if not results:
        st.info("No results found.")
        return

    cols = st.columns(5)
    for i, (img_id, score) in enumerate(results):
        col = cols[i % 5]
        img_path = _resolve_image_path(data_dir, img_id)
        if img_path.exists():
            col.image(
                str(img_path),
                caption=f"{img_id}\nscore={score:.3f}",
                use_container_width=True,
            )
        else:
            col.warning(f"Missing file: {img_id}")


def _encode_query_cluster(query: str) -> np.ndarray:
    _ensure_local_package_path()
    from person_of_intrest import sv_ray_cluster_api

    model_name = sv_ray_cluster_api._get_text_embedding_model()
    vecs = sv_ray_cluster_api.embed_image(
        data_images=[],
        data_texts=[query],
        model_name=model_name,
        batch_size=1,
    )
    return np.asarray(vecs, dtype=np.float32)


def main() -> None:
    st.set_page_config(page_title="Person of Interest", layout="wide")
    st.title("Person of Interest")
    st.markdown(
        "Describe someone in natural language. We search the face corpus and show closest matches."
    )

    # Load index
    try:
        ann_index, id_list, _manifest = index.load_index(
            index.get_index_dir("cluster")
        )
    except FileNotFoundError:
        st.error("No index found.")
        st.info(
            "Build an index after placing images in `data/faces/`:\n\n"
            "- `python scripts/build_index.py`"
        )
        return
    except Exception as e:
        st.error(f"Could not load index: {e}")
        return

    data_dir = data.get_data_dir()

    query = st.text_area(
        "Describe the person of interest",
        placeholder="e.g. A woman with kind eyes and glasses",
    )

    k = st.slider("Number of results", 1, 50, 20)
    st.caption("Embedding backend used: `cluster`")

    if st.button("Search"):
        if not query.strip():
            st.warning("Enter a description to search.")
            return

        with st.spinner("Encoding query and searching..."):
            try:
                # Step 1: encode
                q_emb = _encode_query_cluster(query.strip())

                # Step 2: retrieve more candidates
                results = index.query_index(ann_index, id_list, q_emb, k=50)

                # Step 3: apply gender filtering (SAFE + WORKING)
                constraints = infer_constraints(query)

                if constraints.gender:
                    # create gender prompts
                    gender_prompts = [ "a face of a woman, feminine features","a face of a man, masculine features"]
                    gender_embs = np.array(sv_ray_cluster_api.embed_image(data_images=[],data_texts=gender_prompts,model_name=sv_ray_cluster_api._get_text_embedding_model(),batch_size=2,),dtype=np.float32,)

                    filtered = []

                    for img_id, score in results:
                        try:
                            idx = id_list.index(img_id)
                            emb = ann_index.reconstruct(idx)

                            female_sim = emb @ gender_embs[0]
                            male_sim = emb @ gender_embs[1]

                            if constraints.gender == "female" and female_sim > male_sim:
                                filtered.append((img_id, score))
                            elif constraints.gender == "male" and male_sim > female_sim:
                                filtered.append((img_id, score))

                        except Exception:
                            continue

                    # fallback if filtering too strict
                    if filtered:
                        results = filtered

                # Step 4: final top-k
                results = results[:k]

            except Exception as e:
                st.error(f"Error during search: {e}")
                return

        st.success(f"Showing top {len(results)} matches")
        _display_results(data_dir, results)


if __name__ == "__main__":
    main()