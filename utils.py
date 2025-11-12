import os
import json
import glob
import numpy as np
import faiss
from typing import List, Dict, Tuple

try:
    import pypdf
    HAS_PDF = True
except Exception:
    HAS_PDF = False

import google.generativeai as genai

EMBED_MODEL = "models/text-embedding-004"


def configure_gemini(api_key: str):
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is empty.")
    genai.configure(api_key=api_key)


def load_text_from_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".txt", ".md", ".csv", ".log"}:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if ext == ".pdf":
        if not HAS_PDF:
            raise RuntimeError("pypdf is not installed but PDF file was provided.")
        reader = pypdf.PdfReader(path)
        parts = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                parts.append("")
        return "\n".join(parts)
    return ""


def load_texts_from_folder(folder: str, include_pdf: bool = True) -> List[Dict]:
    exts = ["*.txt", "*.md", "*.csv", "*.log"]
    if include_pdf:
        exts.append("*.pdf")
    paths = []
    for pattern in exts:
        paths.extend(glob.glob(os.path.join(folder, "**", pattern), recursive=True))
    docs = []
    for p in sorted(set(paths)):
        try:
            text = load_text_from_file(p)
            if text.strip():
                docs.append({"path": p, "text": text})
        except Exception:
            continue
    return docs


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    if chunk_size <= 0:
        chunk_size = 1000
    if overlap >= chunk_size:
        overlap = 0
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    return chunks


def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 768), dtype="float32")
    results = []
    for t in texts:
        resp = genai.embed_content(model=EMBED_MODEL, content=t)
        vec = np.array(resp["embedding"], dtype="float32")
        results.append(vec)
    return np.vstack(results)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    if embeddings.size == 0:
        return faiss.IndexFlatL2(768)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, metadatas: List[Dict], base_dir: str):
    os.makedirs(base_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(base_dir, "index.faiss"))
    with open(os.path.join(base_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(metadatas, f, ensure_ascii=False)


def load_index(base_dir: str) -> Tuple[faiss.Index, List[Dict]]:
    index_path = os.path.join(base_dir, "index.faiss")
    meta_path = os.path.join(base_dir, "meta.json")
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Index not found. Run ingestion first.")
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadatas = json.load(f)
    return index, metadatas


def search(index: faiss.Index, metadatas: List[Dict], query_vec: np.ndarray, k: int = 5) -> List[Dict]:
    D, I = index.search(query_vec.astype("float32"), k)
    out = []
    for pos, (dist, idx) in enumerate(zip(D[0].tolist(), I[0].tolist())):
        if idx < 0 or idx >= len(metadatas):
            continue
        item = metadatas[idx].copy()
        item["rank"] = pos + 1
        item["distance"] = float(dist)
        out.append(item)
    return out
