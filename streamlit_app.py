import os
import io
import streamlit as st
from typing import List

from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from langchain_google_genai import ChatGoogleGenerativeAI


# ---------------------------
# Helpers
# ---------------------------
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages_text = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            pages_text.append(f"[Page {i+1}]\n{text}")
    return "\n\n".join(pages_text)


def chunk_text_to_documents(text: str, source_name: str) -> List[Document]:
    base_doc = Document(page_content=text, metadata={"source": source_name})
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    return splitter.split_documents([base_doc])


def build_vectorstore(all_docs: List[Document]):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(all_docs, embedding=embeddings)


def get_llm(temperature: float):
    if "GOOGLE_API_KEY" not in st.secrets:
        st.error("❌ GOOGLE_API_KEY not found in Streamlit secrets.")
        st.stop()

    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=temperature,
    )


def answer_question(llm, retriever, question: str) -> str:
    docs = retriever.invoke(question)


    context = "\n\n".join(
        [f"Source: {d.metadata.get('source','unknown')}\n{d.page_content}" for d in docs]
    )

    prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer (max 5 lines):
""".strip()

    response = llm.invoke(prompt)
    return getattr(response, "content", str(response))


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Document Genie - Gemini RAG", layout="wide")
st.title("📄 Document Genie (Gemini + FAISS)")
st.write("Upload PDFs and ask questions based on their content.")

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3, 0.05)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_chunks = []

    for file in uploaded_files:
        pdf_bytes = file.read()
        text = extract_text_from_pdf_bytes(pdf_bytes)

        if not text.strip():
            st.warning(f"⚠️ No extractable text in: {file.name} (maybe scanned PDF).")
            continue

        chunks = chunk_text_to_documents(text, file.name)
        all_chunks.extend(chunks)

    if all_chunks:
        with st.spinner("Building FAISS index..."):
            st.session_state.vectorstore = build_vectorstore(all_chunks)

        st.success(f"✅ FAISS index built using {len(all_chunks)} chunks.")
    else:
        st.error("❌ Could not extract text from PDFs. Upload text-based PDFs.")

st.divider()

question = st.text_input("Ask your question:")

if st.button("Ask"):
    if not question.strip():
        st.warning("Type a question first.")
    elif st.session_state.vectorstore is None:
        st.warning("Upload PDFs first.")
    else:
        llm = get_llm(temperature)
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})

        with st.spinner("Thinking..."):
            answer = answer_question(llm, retriever, question)

        st.markdown("### ✅ Answer")
        st.write(answer)
