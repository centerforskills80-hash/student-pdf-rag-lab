# Student PDF RAG Lab (Free)

A beginner-friendly Retrieval-Augmented Generation (RAG) demo. Upload a PDF, create a searchable index of its text, and ask questions about the document.

## Free tools used

- **Streamlit Community Cloud** — free app hosting (subject to platform limits)
- **google/flan-t5-small** — small open-source language model with Apache 2.0 license
- **scikit-learn TF-IDF** — free, local text retrieval
- **pypdf** — extracts selectable text from PDF files

No paid API, OpenAI key, or credit card is required. There is no unlimited-token guarantee: hosting has resource limits and the model has a finite context/output length.

## Project files

- `app.py` — Streamlit web interface
- `rag_engine.py` — PDF extraction, chunking, retrieval, and answer generation
- `requirements.txt` — Python packages needed for local/cloud installation

## Run locally on Windows

Python 3.11 is recommended.

1. Download or clone this repository.
2. Open Command Prompt / VS Code terminal in the project folder.
3. Create a virtual environment:

   ```bat
   py -3.11 -m venv .venv
   ```

4. Activate it:

   ```bat
   .venv\Scripts\activate
   ```

5. Install dependencies:

   ```bat
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. Start the app:

   ```bat
   streamlit run app.py
   ```

7. Your browser should open the local app. If not, open the `Local URL` shown in the terminal, usually `http://localhost:8501`.

The first run downloads the model weights, so it may take a while and use several hundred MB of disk space.

## Upload to GitHub

1. Sign in to GitHub and create a new repository, for example `student-pdf-rag-lab`.
2. Keep the repository public if you want the simplest free deployment workflow.
3. Upload `app.py`, `rag_engine.py`, `requirements.txt`, and `README.md` to the repository root.
4. Do **not** upload private PDFs, passwords, API keys, virtual environments, or downloaded model files.

## Deploy to Streamlit Community Cloud

1. Visit https://share.streamlit.io/ and sign in with GitHub.
2. Choose **Create app**.
3. Select your repository, branch (usually `main`), and main file path `app.py`.
4. Select a supported Python version (Python 3.11 is a good choice for this lab).
5. Click **Deploy** and wait for dependency installation and the first model download.
6. Open the generated `*.streamlit.app` URL and test it with a small, text-based PDF.

Official deployment guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

## How the RAG pipeline works

1. **Extract:** `pypdf` reads selectable text from the uploaded PDF.
2. **Chunk:** text is split into overlapping word chunks.
3. **Index:** TF-IDF converts chunks into numerical text vectors.
4. **Retrieve:** cosine similarity selects passages most relevant to the question.
5. **Generate:** retrieved passages and the question are passed to FLAN-T5.
6. **Show evidence:** the interface displays retrieved passages so the student can verify the answer.

## Important limitations

- This starter app does **not** train or fine-tune the language model. RAG retrieves relevant content at question time.
- Scanned PDFs that contain only images need OCR; OCR is not included.
- TF-IDF is a simple lexical retrieval method. It is useful for a first lab but is not as semantically capable as embedding-based retrieval.
- FLAN-T5-small is intentionally lightweight and may produce weak answers for complex questions or long documents.
- Free hosting resources, cold starts, model downloads, and request capacity are limited and can change. This is not an unlimited-token service.
- Uploaded documents are processed by the app's server while the app is running. Do not upload confidential or sensitive documents to a public demo.

## Troubleshooting

**`ModuleNotFoundError` locally:** activate your virtual environment and run `python -m pip install -r requirements.txt`.

**App fails during cloud build:** open the app's logs in Streamlit Community Cloud. Dependency resolution and memory availability can change; try redeploying or simplifying dependencies if the platform reports resource errors.

**PDF has no text:** test whether you can select and copy text from it. If not, it is likely scanned and needs OCR.

**Slow first response:** the model weights are downloaded on first use. Later requests may be faster while the app remains active.
