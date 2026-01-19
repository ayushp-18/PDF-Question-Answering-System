import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")
st.title("📄 Conversational PDF Chatbot (OpenAI + RAG)")
st.write("Upload one or more PDFs and ask questions. The assistant answers from your PDF content.")

temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)

# ---------------------------
# API Key from Streamlit Secrets
# ---------------------------
if "OPENAI_API_KEY" not in st.secrets:
    st.error("OPENAI_API_KEY is missing in Streamlit secrets. Add it in Settings → Secrets.")
    st.stop()

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# LLM + Embeddings
llm = ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# Session ID (optional)
session_id = st.text_input("Session ID", value="default_session")

# Stateful chat store
if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]


# Upload PDFs
uploaded_documents = st.file_uploader(
    "Upload PDF file(s)",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_documents:
    documents = []

    for uploaded_document in uploaded_documents:
        # safer temp file for Streamlit Cloud
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_document.getvalue())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        documents.extend(docs)

        # cleanup temp file
        try:
            os.remove(tmp_path)
        except:
            pass

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Vector store
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Retrievers (Hybrid: BM25 + FAISS)
    faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 4

    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5],
    )

    # History aware retriever prompt
    context_system_prompt = (
        "Given a chat history and the latest user question which might reference context "
        "in the chat history, rewrite it as a standalone question."
        "Do NOT answer the question. Only rewrite it if needed."
    )

    context_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", context_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_prompt)

    # QA prompt
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Answer ONLY using the provided context. "
        "If you don't know the answer, say you don't know.\n\n"
        "Context:\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # Chat input
    user_input = st.text_input("Ask a question from your PDFs:")

    if user_input:
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        st.write("### ✅ Answer")
        st.write(response["answer"])

else:
    st.info("Upload PDF(s) to start chatting.")
