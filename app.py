import os
import streamlit as st
from dotenv import load_dotenv
import numpy as np
import google.generativeai as genai
import io
import faiss

try:
    import pypdf
    _HAS_PDF = True
except Exception:
    _HAS_PDF = False

from utils import configure_gemini, load_index, search, embed_texts, save_index, chunk_text

MODEL_NAME = "gemini-1.5-flash"
MODEL_OPTIONS = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
]


def ensure_config():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY", "")
    index_dir = os.getenv("RAG_INDEX_DIR", ".rag")
    if not api_key:
        st.error("Missing GOOGLE_API_KEY. Add it to .env.")
    return api_key, index_dir


def list_supported_models() -> list:
    try:
        models = genai.list_models()
    except Exception:
        return []
    supported = []
    for m in models:
        try:
            name = getattr(m, "name", "")
            methods = set(getattr(m, "supported_generation_methods", []) or [])
            if not name:
                continue
            # Only keep models that support generateContent
            if "generateContent" in methods:
                supported.append(name)
        except Exception:
            continue
    # Prefer plain gemini names first, then others, sorted
    supported = sorted(supported, key=lambda n: ("gemini" not in n, n))
    return supported


def embed_query(q: str) -> np.ndarray:
    vec = embed_texts([q])
    if vec.ndim == 1:
        vec = vec.reshape(1, -1)
    return vec


def _generate_with_model_name(model_name: str, prompt: str) -> str:
    # Try calling as-is, then try the alternate prefix form
    tried = []
    candidates = [model_name]
    if model_name.startswith('models/'):
        candidates.append(model_name.split('/', 1)[1])
    else:
        candidates.append(f'models/{model_name}')
    for cand in candidates:
        if cand in tried:
            continue
        tried.append(cand)
        try:
            model = genai.GenerativeModel(cand)
            resp = model.generate_content(prompt)
            return resp.text if hasattr(resp, 'text') else str(resp)
        except Exception as e:
            last_err = e
            continue
    raise last_err


def generate_answer(contexts, question, model_name: str):
    context_text = "\n\n".join([f"[Source: {c['source']} | Chunk {c['chunk']}]\n{c['text']}" for c in contexts])
    prompt = (
        "You are a helpful assistant. Answer the question using ONLY the context below.\n"
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n"
    )
    return _generate_with_model_name(model_name, prompt)


def main():
    st.set_page_config(page_title="Local RAG with Gemini", page_icon="🤖", layout="wide")
    st.title("Local Files Q&A (Gemini)")

    api_key, index_dir = ensure_config()
    if api_key:
        genai.configure(api_key=api_key)

    # Auto-refresh available models on first load
    if "available_models" not in st.session_state:
        st.session_state["available_models"] = list_supported_models()

    with st.sidebar:
        st.header("Index")
        st.write("Run ingestion with: `python ingest.py <folder> --include_pdf`. Or upload files below.")
        st.write(f"Index dir: {index_dir}")
        if st.button("Refresh models"):
            st.session_state["available_models"] = list_supported_models()
            if not st.session_state["available_models"]:
                st.warning("No models listed or API not permitted. Using defaults.")
        model_options = st.session_state["available_models"] or MODEL_OPTIONS
        selected_model = st.selectbox("Model", options=model_options, index=0)
        if st.button("Reload Index"):
            st.session_state["reload_index"] = True
        if st.button("Test model"):
            try:
                _ = _generate_with_model_name(selected_model, "Say 'ok'.")
                st.success(f"Model '{selected_model}' works.")
            except Exception as e:
                st.error(f"Selected model failed: {e}")
        st.divider()
        st.subheader("Upload & Ingest")
        chunk_size = st.number_input("Chunk size", min_value=200, max_value=4000, value=1000, step=100)
        overlap = st.number_input("Overlap", min_value=0, max_value=1000, value=200, step=50)
        uploader = st.file_uploader(
            "Upload documents",
            type=["txt", "md", "csv", "log", "pdf"],
            accept_multiple_files=True,
        )
        ingest_clicked = st.button("Ingest uploads")

    if "reload_index" not in st.session_state:
        st.session_state["reload_index"] = False

    try:
        index, metas = load_index(index_dir)
    except Exception as e:
        st.info("Index not found yet. You can upload files in the sidebar to create it.")
        index = faiss.IndexFlatL2(768)
        metas = []

    if 'ingested_count' not in st.session_state:
        st.session_state['ingested_count'] = 0

    def _extract_text_from_upload(uploaded_file):
        name = uploaded_file.name.lower()
        data = uploaded_file.read()
        if any(name.endswith(ext) for ext in [".txt", ".md", ".csv", ".log"]):
            try:
                return data.decode('utf-8', errors='ignore')
            except Exception:
                return ""
        if name.endswith('.pdf'):
            if not _HAS_PDF:
                st.error("pypdf not installed; cannot parse PDF.")
                return ""
            try:
                reader = pypdf.PdfReader(io.BytesIO(data))
                parts = []
                for page in reader.pages:
                    try:
                        parts.append(page.extract_text() or "")
                    except Exception:
                        parts.append("")
                return "\n".join(parts)
            except Exception:
                return ""
        return ""

    if 'uploader' in locals() and ingest_clicked:
        if not uploader:
            st.info("No files uploaded.")
        else:
            new_chunks = []
            new_metas = []
            for f in uploader:
                text = _extract_text_from_upload(f)
                if not text.strip():
                    continue
                chunks = chunk_text(text, chunk_size=int(chunk_size), overlap=int(overlap))
                for i, ch in enumerate(chunks):
                    new_chunks.append(ch)
                    new_metas.append({"source": f"uploaded:{f.name}", "chunk": i, "text": ch})
            if new_chunks:
                embs = embed_texts(new_chunks)
                if embs.ndim == 1:
                    embs = embs.reshape(1, -1)
                index.add(embs.astype("float32"))
                metas.extend(new_metas)
                try:
                    save_index(index, metas, index_dir)
                    st.success(f"Ingested {len(new_chunks)} chunks from {len(new_metas)} entries. Index saved.")
                    st.session_state['ingested_count'] += len(new_chunks)
                except Exception as e:
                    st.error(f"Failed to save index: {e}")
            else:
                st.info("No text extracted from uploaded files.")

    q = st.text_input("Ask a question about your files")
    topk = st.slider("Top-K", min_value=1, max_value=10, value=5)

    if st.button("Search") and q.strip():
        qvec = embed_query(q)
        results = search(index, metas, qvec, k=topk)
        if not results:
            st.info("No results.")
            return
        with st.expander("Retrieved Context", expanded=False):
            for r in results:
                st.markdown(f"- **{r['source']}** (chunk {r['chunk']}, dist {r['distance']:.4f})")
        try:
            answer = generate_answer(results, q, selected_model)
        except Exception as e:
            # Surface what we tried to help troubleshoot
            tries = [selected_model]
            tries.append(selected_model.split('/', 1)[1] if selected_model.startswith('models/') else f"models/{selected_model}")
            st.error(f"Model call failed for '{selected_model}'. Tried: {tries}. Error: {e}")
            st.info("Try a different model in the sidebar (e.g., gemini-1.5-flash-8b or gemini-1.5-pro).")
            return
        st.subheader("Answer")
        st.write(answer)


if __name__ == "__main__":
    main()
