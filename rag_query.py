import json, faiss, numpy as np
from build_index import *


def carregar_indice(in_dir="storage"):
    index = faiss.read_index(f"{in_dir}/faiss.index")
    meta = json.loads(Path(f"{in_dir}/meta.json").read_text(encoding="utf-8"))
    return index, meta

def recuperar(query, k=5):
    index, meta = carregar_indice()
    qv = embeddar_textos([query])  # (1, d)
    D, I = index.search(qv, k)     # distâncias e índices
    resultados = []
    for rank, (score, idx) in enumerate(zip(D[0], I[0]), 1):
        p = meta[idx]
        resultados.append({
            "rank": rank,
            "score": float(score),
            "texto": p["texto"],
            "fonte": p["fonte"],
            "idx_local": p["idx_local"],
        })
    return resultados

