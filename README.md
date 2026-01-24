## 📜 **Semantic & Narrative Analysis of Shri Ramcharitmanas**

### **A Low-Resource NLP System for Conceptual, Narrative, and Cultural Exploration**

## **Project Overview**

### **This project builds an interpretable semantic and narrative analysis system for Shri Ramcharitmanas, a classical Awadhi–Sanskrit epic, using distributional word representations and verse-level embeddings.**

**Rather than treating the text as a flat corpus, the system models:**

- Conceptual meaning (e.g. धर्म, त्याग, भक्ति)
- Narrative progression across Kandas
- Semantic cohesion and transition
- Verse-level similarity and explanation
- The result is an interactive semantic assistant that allows users to explore meaning, not just keywords.

## **Motivation**

### **Classical Indic texts present unique challenges for NLP:**

- Low-resource languages (Awadhi, Sanskritized Hindi)
- Rich morphology and compounding
- Poetic structure and non-literal semantics
- Narrative meaning spread across verses and episodes

#### **How do meaning, concepts, and narrative roles evolve across the epic?**

### **What This Project Does**
- **✔ Concept-based verse retrieval**

- Find verses that express an idea, even if the word itself does not appear.

**Example:**

- Searching for त्याग retrieves verses about renunciation, detachment, and moral restraint across different Kandas.

- **✔ Verse-to-verse semantic similarity**

- Explore how verses relate within and across narrative phases.
- Local reinforcement (same Kand)
- Long-range semantic echoes (different Kand)
- Narrative transitions

- **✔ Narrative-aware explanations**

- Each result is accompanied by a human-readable explanation such as:
- Reinforcement within the same narrative phase
- Philosophical reflection in later Kandas
- Thematic transition near narrative boundaries
  
- ** ✔ Interactive exploration tool**

**A lightweight Streamlit app allows users to:**

- Explore concepts
- Compare verses
- Understand why a verse was retrieved

#### **Core Ideas Behind the System**
##### **1.Verse-level semantics**

- Each verse is represented as a single semantic vector by averaging its token embeddings.
This allows verses—not just words—to be compared meaningfully.

##### **2.Narrative structure matters**

- The epic is explicitly modeled using Kanda boundaries, enabling:
- Intra-Kanda cohesion analysis
- Boundary similarity measurement
- Narrative phase interpretation

##### **3.Interpretability over black-box models**

- Instead of opaque transformers, the system uses FastText embeddings, enabling:
- Subword awareness
- Robust handling of inflections and compounds
- Transparent semantic behavior

## **System Architecture**
┌──────────────┐
│ Raw Text     │
│ (Kanda-wise) │
└──────┬───────┘
       ↓
┌────────────────────┐
│ Preprocessing       │
│ - Unicode normalize │
│ - Verse segmentation│
│ - Tokenization      │
└──────┬─────────────┘
       ↓
┌────────────────────┐
│ Embedding Training  │
│ - Word2Vec (analysis)
│ - FastText (primary)│
└──────┬─────────────┘
       ↓
┌────────────────────┐
│ Verse Embeddings    │
│ (mean pooling)      │
└──────┬─────────────┘
       ↓
┌─────────────────────────────┐
│ Semantic Retrieval Engine   │
│ - Concept search            │
│ - Verse similarity          │
│ - Kand filtering            │
└──────┬──────────────────────┘
       ↓
┌─────────────────────────────┐
│ Narrative Explanation Layer │
│ - Reinforcement             │
│ - Reflection                │
│ - Transition                │
└──────┬──────────────────────┘
       ↓
┌────────────────────┐
│ Streamlit App       │
│ (Interactive UI)    │
└────────────────────┘

## **📂 Repository Structure**
ramcharitmanas_app/
│
├── app.py                      # Streamlit application
│
├── embeddings/
│   ├── verse_embeddings_ft.npy # Verse-level FastText embeddings
│   └── ft_model.gensim         # Trained FastText model
│
├── data/
│   ├── verses_metadata.json    # Verse text + Kand labels
│   └── kanda_indices.json      # Narrative boundaries
│
├── utils/
│   ├── retrieval.py            # Semantic search functions
│   └── explanation.py          # Narrative explanation logic
│
└── README.md

## **Models Used**

🔹 **Word2Vec**

Used for controlled experiments and comparison
Highlights semantic fragmentation and vocabulary sparsity

🔹 **FastText (Primary Model)**

Subword-aware embeddings

Handles:
- Inflections
- Compounds
- Rare poetic vocabulary
- Chosen for production and user-facing features

**Design principle:**
- Use Word2Vec to learn. Use FastText to build.

## **Key Findings & Insights**

- Semantic cohesion increases in later Kandas, especially Uttara Kand.
- Concepts like त्याग shift:
- Ayodhya → moral choice
- Aranya → lived detachment
- Uttara → philosophical resolution
- FastText preserves lexical identity across inflected forms, while Word2Vec fragments them.
- Narrative meaning is often captured without keyword overlap, validating semantic modeling.

## **Evaluation Strategy**

### In the absence of gold labels:

- Quantitative
  - Intra-Kanda cohesion scores
  - Boundary similarity analysis
  - Cross-model comparison

- Qualitative
  - Concept-based case studies
  - Verse-level interpretation
  - Error analysis and failure cases

**Interpretation always accompanies metrics.**

🖥️ How to Run the App
1️⃣ Install dependencies
``pip install streamlit gensim numpy scikit-learn``

2️⃣ Activate environment and run
``streamlit run app.py``

3️⃣ Open browser
``http://localhost:8501``

## **⚠️ Limitations**

- Semantic similarity ≠ doctrinal equivalence
- Scores are relative, not absolute
- No supervised ground truth
- Embeddings reflect corpus bias

### These limitations are explicitly surfaced in the app.

## **Future Work**

- Transformer-based comparison (IndicBERT / MuRIL)
- Cross-text alignment with Valmiki Ramayana
- Concept heatmaps across narrative phases
- FAISS indexing for large-scale retrieval
- Multilingual extension (Sanskrit ↔ Hindi)

## **What This Project Demonstrates**

- Low-resource NLP competence
- Research-oriented thinking
- Narrative and cultural sensitivity
- Interpretable ML system design

### **Ability to turn analysis into a usable tool**

## **Final Note**

#### => This project is not about finding the “best model.”
#### => It is about understanding how meaning is constructed, expressed, and transformed in a classical epic.
