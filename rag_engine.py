"""
Simple PDF RAG engine for a student lab.

RAG does not fine-tune/train the language model. It extracts and chunks PDF text,
retrieves relevant chunks for each question, then asks FLAN-T5 to answer from them.
"""
from __future__ import annotations

from typing import List, Dict
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_NAME = "google/flan-t5-small"


def extract_pdf_text(pdf_file) -> str:
    """Extract text from an uploaded PDF-like file object."""
    reader = PdfReader(pdf_file)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {page_number}]\n{text}")
    full_text = "\n\n".join(pages).strip()
    if not full_text:
        raise ValueError(
            "No selectable text was found. This may be a scanned/image-only PDF. "
            "Try a PDF with selectable text; OCR is not included in this starter lab."
        )
    return full_text


def split_into_chunks(text: str, chunk_words: int = 180, overlap_words: int = 35) -> List[str]:
    """Split text into word chunks with a small overlap."""
    words = text.split()
    if not words:
        return []
    chunk_words = max(50, int(chunk_words))
    overlap_words = max(0, min(int(overlap_words), chunk_words - 1))
    step = chunk_words - overlap_words
    chunks = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_words]).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_words >= len(words):
            break
    return chunks


class SimpleRAG:
    """TF-IDF retrieval plus a small open-source text-generation model."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.chunks: List[str] = []
        self.vectorizer = None
        self.chunk_vectors = None

    def build_index(self, text: str, chunk_words: int = 180, overlap_words: int = 35) -> int:
        self.chunks = split_into_chunks(text, chunk_words, overlap_words)
        if not self.chunks:
            raise ValueError("The PDF did not contain enough readable text.")
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.chunk_vectors = self.vectorizer.fit_transform(self.chunks)
        return len(self.chunks)

    def _load_model(self):
        if self.tokenizer is None or self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.eval()

    def retrieve(self, question: str, top_k: int = 3) -> List[Dict]:
        if self.vectorizer is None or self.chunk_vectors is None:
            raise RuntimeError("Upload and index a PDF first.")
        question_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(question_vector, self.chunk_vectors).ravel()
        best_indices = scores.argsort()[::-1][:max(1, min(int(top_k), len(self.chunks)))]
        return [
            {"text": self.chunks[int(i)], "score": float(scores[int(i)])}
            for i in best_indices
        ]

    def answer(self, question: str, top_k: int = 3, max_new_tokens: int = 128) -> Dict:
        if not question.strip():
            raise ValueError("Please enter a question.")
        retrieved = self.retrieve(question, top_k=top_k)
        context = "\n\n".join(
            f"Passage {i + 1}: {item['text']}"
            for i, item in enumerate(retrieved)
        )
        prompt = (
            "Answer the question using only the context below. "
            "If the context does not contain the answer, say: "
            "'I could not find that information in the uploaded PDF.' "
            "Do not make up facts.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )
        self._load_model()
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max(16, min(int(max_new_tokens), 200)),
                num_beams=2,
                early_stopping=True,
            )
        answer_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        return {"answer": answer_text, "sources": retrieved}
