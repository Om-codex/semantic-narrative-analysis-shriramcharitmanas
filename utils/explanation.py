def explain_concept_result(concept, verse):
    kanda = verse["kanda"]

    if kanda == "Bal":
        return f"In Bal Kand, {concept} appears in formative and relational contexts."

    if kanda == "Ayodhya":
        return f"Here {concept} is framed as moral duty within social order."

    if kanda == "Aranya":
        return f"This verse expresses {concept} through lived detachment and ascetic experience."

    if kanda == "Kishkindha":
        return f"{concept} emerges through alliance, loyalty, and collective action."

    if kanda == "Sundar":
        return f"{concept} is infused with hope, devotion, and perseverance."

    if kanda == "Lanka":
        return f"{concept} is tested in confrontation and moral conflict."

    if kanda == "Uttara":
        return f"In Uttara Kand, {concept} is philosophically resolved and moralized."

    return f"This verse provides contextual grounding for {concept}."
