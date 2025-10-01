from sentence_transformers import SentenceTransformer
import numpy as np
import faiss, json

import numpy as np
from pathlib import Path
import pandas as pd


def carregar_documentos_txt(pasta="data"):
    docs = []
    for p in Path(pasta).glob("**/*.txt"):
        docs.append({"conteudo": p.read_text(encoding="utf-8"), "fonte": str(p)})
    return docs

def chunk_texto(txt, max_chars=1000, overlap=150):
    chunks = []
    start = 0
    n = len(txt)
    while start < n:
        end = min(start + max_chars, n)
        if end == n: break
        chunk = txt[start:end]
        chunks.append(chunk.strip())
        start = end - overlap  # recua um pouco para sobrepor
        if start < 0: start = 0
    return [c for c in chunks if c]

def gerar_passagens(docs):
    passagens = []
    for d in docs:
        partes = chunk_texto(d["conteudo"])
        for i, ch in enumerate(partes):
            passagens.append({
                "texto": ch,
                "fonte": d["fonte"],
                "idx_local": i
            })
    return passagens

modelo_emb = SentenceTransformer("all-MiniLM-L6-v2")

def embeddar_textos(textos):
    # retorna matriz (n, d) em float32
    v = modelo_emb.encode(textos, convert_to_numpy=True, normalize_embeddings=True)
    return v.astype("float32")

def construir_indice(passagens, out_dir="storage"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    textos = [p["texto"] for p in passagens]
    embs = embeddar_textos(textos)  # (n, d)

    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)     # IP = produto interno (use vetores normalizados)
    index.add(embs)

    faiss.write_index(index, f"{out_dir}/faiss.index")

    # salve metadados para reconstruir citações
    with open(f"{out_dir}/meta.json", "w", encoding="utf-8") as f:
        json.dump(passagens, f, ensure_ascii=False, indent=2)

# Uso:
# docs = carregar_documentos_txt("data")
# passagens = gerar_passagens(docs)
# print(passagens)
# construir_indice(passagens)

