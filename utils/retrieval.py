import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cosine_sim(a, b):
    return cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0]


def get_similar_verses(
    verse_id,
    embeddings,
    verses,
    top_k=5,
    kanda_filter=None
):
    query_vec = embeddings[verse_id]
    sims = []

    for i, vec in enumerate(embeddings):
        if i == verse_id:
            continue

        if kanda_filter and verses[i]["kanda"] != kanda_filter:
            continue

        sim = cosine_sim(query_vec, vec)
        sims.append((i, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]


def search_by_concept(
    concept_vector,
    embeddings,
    verses,
    top_k=5,
    kanda_filter=None
):
    sims = []

    for i, vec in enumerate(embeddings):
        if kanda_filter and verses[i]["kanda"] != kanda_filter:
            continue

        sim = cosine_sim(concept_vector, vec)
        sims.append((i, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]
