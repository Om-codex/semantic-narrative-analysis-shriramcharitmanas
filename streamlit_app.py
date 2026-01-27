from huggingface_hub import hf_hub_download, snapshot_download
import streamlit as st
import json
import numpy as np
import os
from gensim.models import FastText

DATASET_REPO = "OmCodex/shriramartha-assets"

from utils.retrieval import (
    get_similar_verses,
    search_by_concept
)
from utils.explanation import explain_concept_result

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3 {
        letter-spacing: -0.02em;
    }
    .verse-box {
        background-color: #111827;
        border-left: 4px solid #6366f1;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 0.5rem;
    }
    .meta {
        color: #9ca3af;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Ramartha — Semantic Explorer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_resources():
    with st.spinner("Loading vector embeddings and model..."):

        # 1. Load verse embeddings (single file → hf_hub_download is correct)
        emb_path = hf_hub_download(
            repo_id=DATASET_REPO,
            filename="embeddings/verse_embeddings_ft.npy",
            repo_type="dataset"
        )
        embeddings = np.load(emb_path, mmap_mode="r")

        # 2. Load FastText model (MULTI-FILE → snapshot_download is mandatory)
        snapshot_dir = snapshot_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            allow_patterns="embeddings/*"
        )

        ft_model_path = os.path.join(
            snapshot_dir,
            "embeddings",
            "ft_model.gensim"
        )
        ft_model = FastText.load(ft_model_path)

        # 3. Load verses metadata
        verses_path = hf_hub_download(
            repo_id=DATASET_REPO,
            filename="data/verses_metadata.json",
            repo_type="dataset"
        )
        with open(verses_path, "r", encoding="utf-8") as f:
            verses = json.load(f)

        # 4. Load kanda indices
        kanda_path = hf_hub_download(
            repo_id=DATASET_REPO,
            filename="data/kanda_indices.json",
            repo_type="dataset"
        )
        with open(kanda_path, "r", encoding="utf-8") as f:
            kanda_indices = json.load(f)

    return embeddings, ft_model, verses, kanda_indices



embeddings, ft_model, verses, kanda_indices = load_resources()
kanda_names = ["All"] + [k[0] for k in kanda_indices]



st.title("📜 Ramartha")
st.caption(
    "A semantic & narrative intelligence system for *Shri Ramcharitmanas* "
    "built using low-resource NLP."
)
st.divider()

tab_concept, tab_verse, tab_about = st.tabs(
    ["🔍 Concept Explorer", "📖 Verse Explorer", "ℹ️ About"]
)


with tab_concept:
    st.subheader("Conceptual Exploration")
    st.write(
        "Explore how a **concept** is expressed across verses and narrative phases."
    )

    # --- Suggested concepts ---
    SUGGESTED_CONCEPTS = [
        "धर्म",
        "त्याग",
        "भक्ति",
        "राम",
        "सीता",
        "हनुमान"
    ]

    left, right = st.columns([2, 1])

    with left:
        selected_concept = st.selectbox(
            "Choose a suggested concept (optional)",
            [""] + SUGGESTED_CONCEPTS,
            index=0
        )

        concept = st.text_input(
            "Or enter your own concept (Devanagari)",
            value=selected_concept,
            placeholder="उदा: करुणा, युद्ध, वन"
        )

    with right:
        kanda_filter = st.selectbox(
            "Narrative scope",
            kanda_names
        )

    # --- Concept search logic ---
    if concept.strip():
        if concept not in ft_model.wv:
            st.warning("This concept is not present in the learned vocabulary.")
        else:
            results = search_by_concept(
                ft_model.wv[concept],
                embeddings,
                verses,
                top_k=5,
                kanda_filter=None if kanda_filter == "All" else kanda_filter
            )

            st.divider()
            st.subheader("Most Relevant Verses")

            for idx, score in results:
                verse = verses[idx]
                explanation = explain_concept_result(concept, verse)

                st.markdown(
                    f"""
                    <div class="verse-box">
                        <div class="meta">
                            {verse['kanda']} Kand · similarity {score:.3f}
                        </div>
                        <div>{" ".join(verse["text"])}</div>
                        <div class="meta">{explanation}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.info("Select a suggested concept or enter your own to begin exploration.")


with tab_verse:
    st.subheader("Verse Similarity Explorer")
    st.write(
        "Select a verse and explore semantically similar passages across the text."
    )

    verse_id = st.slider(
        "Verse ID",
        min_value=0,
        max_value=len(verses) - 1,
        value=100
    )

    base_verse = verses[verse_id]

    st.markdown(
        f"""
        <div class="verse-box">
            <div class="meta">Query Verse · {base_verse['kanda']} Kand</div>
            <div>{" ".join(base_verse["text"])}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Find Similar Verses"):
        results = get_similar_verses(
            verse_id,
            embeddings,
            verses,
            top_k=5
        )

        st.subheader("Related Verses")

        for idx, score in results:
            verse = verses[idx]
            st.markdown(
                f"""
                <div class="verse-box">
                    <div class="meta">{verse['kanda']} Kand · similarity {score:.3f}</div>
                    <div>{" ".join(verse["text"])}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
with tab_about:
    st.subheader("About Ramartha")

    st.markdown("""
**Ramartha** is a research-oriented system for semantic and narrative analysis of
*Shri Ramcharitmanas* in a low-resource NLP setting.

### Core ideas
- Verse-level semantic representations  
- Narrative-phase (Kand) aware analysis  
- Subword modeling for poetic vocabulary  

### What this tool does
✔ Explores meaning, not keywords  
✔ Supports interpretation, not doctrine  
✔ Makes assumptions explicit  

### What this tool does not do
✘ Claim theological authority  
✘ Replace close reading  
✘ Perform exact matching  

### Technical notes
- FastText embeddings (offline trained)
- Hugging Face Hub for model hosting
- Streamlit for interpretability-first UI
""")
