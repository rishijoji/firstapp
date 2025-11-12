# Local Files Q&A with Gemini (Free API)

## Setup

- Install Python 3.10+
- Create and activate a virtual environment.
- Install dependencies:

```
pip install -r requirements.txt
```

- Copy `.env.example` to `.env` and set `GOOGLE_API_KEY`.
- Optional: change `RAG_INDEX_DIR` (default `.rag`).

## Ingest your folder

```
python ingest.py "C:\\Path\\To\\Your\\Folder" --include_pdf
```

- Supported: .txt, .md, .csv, .log, and PDFs when `--include_pdf` is used.
- Creates FAISS index + metadata in `.rag/`.

## Run the app

```
streamlit run app.py
```

- Open the URL printed by Streamlit.
- Ask questions. The app retrieves top-k chunks from your index and calls Gemini to answer using only that context.

## Notes

- Requires a free Gemini API key from Google AI Studio.
- Keep your `.env` private.
