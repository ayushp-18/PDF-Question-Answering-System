"""
Streamlit app: PDF Question Answering Chat

Usage:
- Provide HF_TOKEN + (GROQ_API_KEY or OPENAI_API_KEY) via Streamlit secrets or environment.
- Run locally: `streamlit run streamlit_app.py`
"""
import io
import os
import streamlit as st
from typing import List

# PDF/Text handling
from pypdf import PdfReader

# LangChain-ish utilities (keep compatibility with your repo imports)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# LLM imports (try to support both Groq and OpenAI)
llm_backend = None
try:
    from langchain_groq import ChatGroq
    llm_backend = "groq"
except Exception:
    pass

try:
    from langchain.chat_models import ChatOpenAI
    if llm_backend is None:
        llm_backend = "openai"
except Exception:
    # sometimes import will fail if package not installed
    if llm_backend is None:
        llm_backend = None

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
        if text:
            # add metadata marker per page
            text_pages.append(f"[Page {i+1}]\n" + text)
    return "\n\n".join(text_pages)

def make_chunks_from_text(text: str, filename: str) -> List[Document]:
    # Create a Document object and let the text splitter create chunks
    initial_doc = Document(page_content=text, metadata={"source": filename})
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents([initial_doc])
    return chunks

def get_embeddings():
    # Expect HF_TOKEN in secrets or environment for HuggingFaceEmbeddings
    hf_token = os.environ.get("HF_TOKEN") or st.secrets.get("HF_TOKEN", None) if "HF_TOKEN" in st.secrets else os.environ.get("HF_TOKEN")
    if not hf_token:
        st.warning("HF_TOKEN not found. HuggingFaceEmbeddings typically needs HF_TOKEN in secrets.")
    # model_name can be adjusted
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", huggingfacehub_api_token=hf_token)

def init_llm(temperature: float = 0.2):
    # Prefer Groq if GROQ_API_KEY present, else fall back to OpenAI if key available.
    groq_key = st.secrets.get("GROQ_API_KEY") if "GROQ_API_KEY" in st.secrets else os.environ.get("GROQ_API_KEY")
    openai_key = st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else os.environ.get("OPENAI_API_KEY")

    if groq_key and llm_backend == "groq":
        llm = ChatGroq(groq_api_key=groq_key, model_name="llama3-70b-8192", temperature=temperature)
        return llm, "groq"
    if openai_key and llm_backend == "openai":
        # model name can be changed by user
        llm = ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo")
        return llm, "openai"
    # No supported llm available
    return None, None

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="PDF Q&A Chat", layout="wide")
st.title("PDF Question-Answering — Upload a PDF and ask questions")
st.write("Upload one or more PDFs, the app will build a searchable vector index, and you can ask questions (chat-like) about the PDF contents.")

# temperature slider
temperature = st.sidebar.slider("LLM temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

# Session management for vectorstore and history
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "docs_metadata" not in st.session_state:
    st.session_state.docs_metadata = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (user, assistant) tuples

# File uploader
uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    # Build or rebuild vector store when new pdfs uploaded
    all_chunks = []
    for uploaded in uploaded_files:
        raw_bytes = uploaded.read()
        text = extract_text_from_pdf_bytes(raw_bytes)
        if not text.strip():
            st.warning(f"No extractable text found in {uploaded.name}. PDF might be scanned/ image-only.")
        chunks = make_chunks_from_text(text, uploaded.name)
        all_chunks.extend(chunks)

    if not all_chunks:
        st.error("No text chunks were created from uploaded PDFs. Check file contents.")
    else:
        with st.spinner("Creating embeddings and building FAISS index (this may take a moment)..."):
            embeddings = get_embeddings()
            # Create/update vector store
            vectorstore = FAISS.from_documents(documents=all_chunks, embedding=embeddings)
            st.session_state.vectorstore = vectorstore
            st.session_state.docs_metadata = {"n_chunks": len(all_chunks)}
        st.success(f"Index built from {len(all_chunks)} chunks.")

# Query box and conversation UI
col1, col2 = st.columns([3, 1])
with col1:
    user_question = st.text_input("Ask a question about the uploaded PDF(s):")
    ask_button = st.button("Ask")

with col2:
    st.write("Conversation history")
    if st.session_state.chat_history:
        for i, (u, a) in enumerate(reversed(st.session_state.chat_history[-10:]), 1):
            st.markdown(f"**You:** {u}")
            st.markdown(f"**Assistant:** {a}")
            st.write("---")
    else:
        st.write("No history yet")

if ask_button:
    if not user_question:
        st.warning("Please enter a question.")
    elif not st.session_state.vectorstore:
        st.warning("Please upload PDF(s) first to build the index.")
    else:
        # init llm
        llm, which = init_llm(temperature=temperature)
        if llm is None:
            st.error("No LLM configured. Set GROQ_API_KEY (for Groq) or OPENAI_API_KEY (for OpenAI) in Streamlit secrets.")
        else:
            # create retriever and chain
            retriever = st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
            with st.spinner("Running query..."):
                result = qa_chain({"query": user_question})
            answer = result["result"] if isinstance(result, dict) else str(result)
            source_docs = result.get("source_documents", []) if isinstance(result, dict) else []

            # show answer
            st.markdown("### Answer")
            st.write(answer)

            # show sources
            if source_docs:
                st.markdown("#### Source snippets (from top documents)")
                for doc in source_docs[:4]:
                    meta = getattr(doc, "metadata", {})
                    src = meta.get("source", "unknown")
                    st.markdown(f"- **Source:** {src}")
                    excerpt = (doc.page_content[:800] + "...") if len(doc.page_content) > 800 else doc.page_content
                    st.text(excerpt)
                    st.write("---")

            # update history
            st.session_state.chat_history.append((user_question, answer))
            # keep history length reasonable
            if len(st.session_state.chat_history) > 50:
                st.session_state.chat_history = st.session_state.chat_history[-50:]
