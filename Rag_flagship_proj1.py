import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="PDF Q&A (OpenAI)", layout="wide")
st.title("📄 PDF Question Answering (OpenAI + FAISS)")

st.write("Upload a PDF and ask questions. The answer will come only from the PDF content.")

# ------------------ OpenAI Key ------------------
if "OPENAI_API_KEY" not in st.secrets:
    st.error("❌ OPENAI_API_KEY not found in Streamlit Secrets.")
    st.stop()

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# ------------------ LLM + Embeddings ------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ------------------ Upload PDFs ------------------
uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_docs = []

    for uploaded_file in uploaded_files:
        # Save to a temp file for PyPDFLoader
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        all_docs.extend(docs)

        # delete temp
        try:
            os.remove(pdf_path)
        except:
            pass

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)

    # Vector store
    vector_store = FAISS.from_documents(chunks, embeddings)

    st.success(f"✅ Loaded {len(all_docs)} pages and created {len(chunks)} chunks.")

    # Ask question
    user_question = st.text_input("Ask your question:")

    if user_question:
        # Retrieve top chunks
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        retrieved_docs = retriever.get_relevant_documents(user_question)

        context = "\n\n".join([d.page_content for d in retrieved_docs])

        prompt = f"""
You are a helpful assistant answering ONLY from the PDF context.

If the answer is not available inside the context, say:
"I don't know based on the PDF."

PDF Context:
{context}

Question: {user_question}

Answer:
"""

        answer = llm.invoke(prompt)

        st.subheader("✅ Answer")
        st.write(answer.content)

        # Optional: show sources
        with st.expander("Show retrieved context"):
            for i, d in enumerate(retrieved_docs, 1):
                st.markdown(f"### Chunk {i}")
                st.write(d.page_content)

else:
    st.info("👆 Upload PDF(s) to start.")
