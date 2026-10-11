import streamlit as st
from pypdf.errors import PdfReadError
from rag_engine import SimpleRAG, extract_pdf_text

st.set_page_config(page_title="Student PDF RAG Lab", page_icon="📚", layout="wide")

st.title("📚 Student PDF RAG Lab")
st.write(
    "Upload a PDF, build a searchable index, and ask questions answered with help "
    "from a free open-source language model."
)

st.info(
    "Learning note: this app does not retrain or fine-tune the language model. "
    "It indexes your PDF, retrieves relevant passages, and supplies them to the model (RAG)."
)

@st.cache_resource(show_spinner="Loading the RAG engine...")
def get_rag_engine():
    return SimpleRAG()

if "rag" not in st.session_state:
    st.session_state.rag = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "indexed" not in st.session_state:
    st.session_state.indexed = False

with st.sidebar:
    st.header("1. Upload your PDF")
    uploaded_pdf = st.file_uploader("Choose a PDF file", type=["pdf"])
    chunk_words = st.slider("Words per text chunk", 80, 300, 180, 20)
    overlap_words = st.slider("Overlap between chunks", 0, 60, 35, 5)
    top_k = st.slider("Relevant passages to retrieve", 1, 5, 3)
    st.caption("First model load may take a few minutes because the model is downloaded.")

    build_clicked = st.button("📄 Read PDF and build index", type="primary", use_container_width=True)

if build_clicked:
    if uploaded_pdf is None:
        st.warning("Please upload a PDF first.")
    else:
        try:
            with st.spinner("Reading PDF and building the search index..."):
                text = extract_pdf_text(uploaded_pdf)
                rag = get_rag_engine()
                number_chunks = rag.build_index(
                    text, chunk_words=chunk_words, overlap_words=overlap_words
                )
                st.session_state.rag = rag
                st.session_state.pdf_name = uploaded_pdf.name
                st.session_state.indexed = True
                st.session_state.document_text_length = len(text)
                st.session_state.number_chunks = number_chunks
            st.success(f"PDF indexed successfully: {number_chunks} text chunks created.")
        except (ValueError, PdfReadError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Could not process this PDF: {exc}")

if st.session_state.indexed and st.session_state.rag is not None:
    st.success(f"Ready to ask questions about: **{st.session_state.pdf_name}**")
    a, b = st.columns(2)
    a.metric("Text chunks indexed", st.session_state.number_chunks)
    b.metric("Extracted characters", f"{st.session_state.document_text_length:,}")

    st.header("2. Ask a question")
    question = st.text_input(
        "Your question",
        placeholder="Example: What are the main concepts explained in this document?",
    )
    if st.button("💬 Get answer", type="primary"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            try:
                with st.spinner("Finding relevant passages and generating an answer..."):
                    result = st.session_state.rag.answer(question, top_k=top_k)
                st.subheader("Answer")
                st.write(result["answer"])
                st.subheader("Retrieved passages (evidence)")
                for i, source in enumerate(result["sources"], start=1):
                    with st.expander(f"Passage {i} — retrieval score {source['score']:.3f}"):
                        st.write(source["text"])
                st.caption(
                    "Retrieval scores are similarity signals, not probabilities. "
                    "Always verify answers against the original PDF."
                )
            except Exception as exc:
                st.error(f"Could not generate an answer: {exc}")
else:
    st.markdown("### How to use")
    st.markdown(
        "1. Upload a PDF in the left sidebar.\n"
        "2. Click **Read PDF and build index**.\n"
        "3. Type a question and click **Get answer**.\n"
        "4. Open the retrieved passages to see what evidence was supplied to the model."
    )

st.divider()
st.caption("Free student demo • TF-IDF retrieval • google/flan-t5-small • No API key required")
