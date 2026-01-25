import streamlit as st
import json
import numpy as np
from gensim.models import FastText

from utils.retrieval import (
    get_similar_verses,
    search_by_concept
)
from utils.explanation import explain_concept_result


# --------------------
# Load Data
# --------------------

@st.cache_resource
def load_resources():
    embeddings = np.load("embeddings/verse_embeddings_ft.npy")
    ft_model = FastText.load("embeddings/ft_model.genism")


    with open("data/verses_metadata.json") as f:
        verses = json.load(f)

    with open("data/kanda_indices.json") as f:
        kanda_indices = json.load(f)
    
    

    return embeddings, ft_model, verses, kanda_indices


embeddings, ft_model, verses, kanda_indices = load_resources()
kanda_names = ["All"] + [k[0] for k in kanda_indices]

# --------------------
# UI
# --------------------

st.set_page_config(page_title="Ramcharitmanas Semantic Explorer", layout="wide")

st.title("📜 Semantic & Narrative Explorer — Shri Ramcharitmanas")
st.markdown(
    "Explore **concepts, verses, and narrative meaning** using computational semantics."
)

tabs = st.tabs(["🔍 Concept Search", "📖 Verse Explorer", "ℹ️ About"])


# --------------------
# TAB 1: Concept Search
# --------------------

with tabs[0]:
    st.header("Concept-based Exploration")

    SUGGESTED_CONCEPTS = [
        "धर्म",
        "त्याग",
        "भक्ति",
        "राम",
        "सीता",
        "हनुमान"
    ]
    selected_concept = st.selectbox(
        "Choose a suggested concept (optional)",
        [""] + SUGGESTED_CONCEPTS
    )

    concept = st.text_input(
        "Or enter your own concept",
        value=selected_concept
    )
    if concept.strip() == "":
        st.info("Please choose or enter a concept to explore.")
        st.stop()
    kanda_filter = st.selectbox(
        "Restrict to Kand (optional)",
        kanda_names
    )

    if concept:
        if concept not in ft_model.wv:
            st.warning("Concept not in vocabulary.")
        else:
            concept_vec = ft_model.wv[concept]

            results = search_by_concept(
                concept_vec,
                embeddings,
                verses,
                top_k=5,
                kanda_filter=None if kanda_filter == "All" else kanda_filter
            )

            for idx, score in results:
                verse = verses[idx]
                explanation = explain_concept_result(concept, verse)

                st.markdown(f"**{verse['kanda']} Kand** — similarity `{score:.3f}`")
                st.write(" ".join(verse["text"]))
                st.caption(explanation)
                st.divider()


# --------------------
# TAB 2: Verse Explorer
# --------------------

with tabs[1]:
    st.header("Verse Similarity Explorer")

    verse_id = st.number_input(
        "Enter Verse ID",
        min_value=0,
        max_value=len(verses) - 1,
        value=100
    )

    if st.button("Find Similar Verses"):
        base_verse = verses[verse_id]
        st.subheader("Query Verse")
        st.write(" ".join(base_verse["text"]))
        st.caption(f"Kand: {base_verse['kanda']}")

        results = get_similar_verses(
            verse_id,
            embeddings,
            verses,
            top_k=5
        )

        st.subheader("Similar Verses")
        for idx, score in results:
            verse = verses[idx]
            st.markdown(f"**{verse['kanda']} Kand** — similarity `{score:.3f}`")
            st.write(" ".join(verse["text"]))
            st.divider()


# --------------------
# TAB 3: About
# --------------------

with tabs[2]:
    st.header("About This Project")

    st.markdown("""
This system performs **semantic and narrative analysis** of *Shri Ramcharitmanas* using:

- FastText embeddings (subword-aware)
- Verse-level semantic representations
- Concept-based retrieval
- Narrative-phase interpretation

### What this tool is:
✔ Meaning-aware  
✔ Interpretable  
✔ Low-resource NLP focused  

### What this tool is NOT:
✘ Keyword search  
✘ Exact matching  
✘ Oracle of truth  

Semantic similarity reflects **textual meaning**, not doctrinal interpretation.
""")
