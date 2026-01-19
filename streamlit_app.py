import os
import io
import streamlit as st
from typing import List

from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# ✅ Gemini
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.chains import RetrievalQA


# ---------------------------
# Helper functions
# ---------------------------
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text_pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            text_pages.append(f"[Page {i+1}]\n{text}")
    return "\n\n".join(text_pages)


def make_chunks_from_text(text: str, filename: str) -> List[Document]:
    initial_doc = Document(page_content=text, metadata={"source": filename})
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    return splitter.split_documents([initial_doc])


def get_embeddings():
    # For all-MiniLM-L6-v2, HF_TOKEN is usually NOT required.
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def get_gemini_llm(temperature: float):
    if "GOOGLE_API_KEY" not in st.secrets:
        st.error("❌ GOOGLE_API_KEY not found in Streamlit secrets.")
        st.stop()

    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=temperature,
    )


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Document Genie - Gemini RAG", layout="wide")
st.title("📄 Document Genie (RAG) — Gemini + FAISS")
st.write("Upload PDFs, build an index, and ask questions from the content.")

temperature = st.sidebar.slider("LLM temperature", 0.0, 1.0, 0.3, 0.05)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_chunks = []

    for uploaded in uploaded_files:
        raw_bytes = uploaded.read()
        text = extract_text_from_pdf_bytes(raw_bytes)

        if not text.strip():
            st.warning(f"⚠️ No extractable text found in {uploaded.name}. It may be scanned.")
            continue

        chunks = make_chunks_from_text(text, uploaded.name)
        all_chunks.extend(chunks)

    if not all_chunks:
        st.error("❌ No text chunks were created. Upload a text-based PDF.")
    else:
        with st.spinner("Building embeddings + FAISS index..."):
            embeddings = get_embeddings()
            st.session_state.vectorstore = FAISS.from_documents(all_chunks, embedding=embeddings)

        st.success(f"✅ Index built successfully from {len(all_chunks)} chunks.")


st.divider()

question = st.text_input("Ask a question:")

if st.button("Ask"):
    if not question.strip():
        st.warning("Enter a question.")
    elif st.session_state.vectorstore is None:
        st.warning("Upload PDFs first.")
    else:
        llm = get_gemini_llm(temperature=temperature)

        retriever = st.session_state.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )

        with st.spinner("Thinking..."):
            result = qa_chain({"query": question})

        answer = result["result"]
        st.markdown("### ✅ Answer")
        st.write(answer)

        st.session_state.chat_history.append((question, answer))

        sources = result.get("source_documents", [])
        if sources:
            st.markdown("### 📌 Sources (snippets)")
            for doc in sources[:3]:
                src = doc.metadata.get("source", "unknown")
                st.markdown(f"**Source:** {src}")
                snippet = doc.page_content[:700]
                st.text(snippet)
                st.write("---")


# Optional chat history display
if st.session_state.chat_history:
    st.sidebar.markdown("## 🕘 Recent Questions")
    for q, a in st.session_state.chat_history[-5:][::-1]:
        st.sidebar.markdown(f"**Q:** {q}")
        st.sidebar.markdown(f"**A:** {a[:120]}...")
        st.sidebar.write("---")
