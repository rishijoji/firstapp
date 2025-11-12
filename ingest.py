import os
import argparse
from dotenv import load_dotenv
import numpy as np

from utils import (
    configure_gemini,
    load_texts_from_folder,
    chunk_text,
    embed_texts,
    build_faiss_index,
    save_index,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Path to folder with documents")
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--include_pdf", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY", "")
    index_dir = os.getenv("RAG_INDEX_DIR", ".rag")

    configure_gemini(api_key)

    docs = load_texts_from_folder(args.folder, include_pdf=args.include_pdf)
    metadatas = []
    all_chunks = []
    for d in docs:
        chunks = chunk_text(d["text"], chunk_size=args.chunk_size, overlap=args.overlap)
        for i, ch in enumerate(chunks):
            metadatas.append({"source": d["path"], "chunk": i, "text": ch})
            all_chunks.append(ch)

    if not all_chunks:
        print("No content found to index.")
        return

    embeddings = embed_texts(all_chunks)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    index = build_faiss_index(embeddings)
    save_index(index, metadatas, index_dir)
    print(f"Indexed {len(all_chunks)} chunks from {len(docs)} files into {index_dir}")


if __name__ == "__main__":
    main()
