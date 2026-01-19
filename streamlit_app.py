import os
import streamlit as st

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# ✅ Gemini LLM
from langchain_google_genai import ChatGoogleGenerativeAI


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Document Genie (RAG)", layout="wide")
st.title("Conversational Chat Bot with PDF upload + Chat History")
st.write("Upload PDFs and ask questions from the content.")


# -------------------------------
# Secrets / Keys
# -------------------------------
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("❌ GOOGLE_API_KEY not found in Streamlit secrets.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# HF_TOKEN is optional unless you need it
if "HF_TOKEN" in st.secrets:
    os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]


# -------------------------------
# Embeddings
# -------------------------------
# Uses local sentence-transformers model (no HF_TOKEN required usually)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# -------------------------------
# LLM (Gemini)
# -------------------------------
temperature = st.slider("Set temperature", 0.0, 1.0, 0.7)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=temperature,
)


# -------------------------------
# Session ID + Storage
# -------------------------------
session_id = st.text_input("Enter your session id", value="default_session")

if "store" not in st.session_state:
    st.session_state.store = {}


# -------------------------------
# Upload PDFs
# -------------------------------
uploaded_documents = st.file_uploader(
    "Choose PDF file(s) to upload",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_documents:
    documents = []

    for uploaded_document in uploaded_documents:
        temp_pdf = "./temp.pdf"
        with open(temp_pdf, "wb") as file:
            file.write(uploaded_document.getvalue())

        loader = PyPDFLoader(temp_pdf)
        docs = loader.load()
        documents.extend(docs)

    # ✅ Chunking (keep your chunk params)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3800,
        chunk_overlap=1000
    )
    chunks = text_splitter.split_documents(documents)

    # ✅ Vector store
    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)

    # ✅ Hybrid retriever
    faiss_semantic_retriever = vector_store.as_retriever()
    bm25_retriever = BM25Retriever.from_documents(documents=chunks)

    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_semantic_retriever],
        weights=[0.5, 0.5]
    )

    # -------------------------------
    # Context prompt (history aware)
    # -------------------------------
    context_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    context_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", context_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, context_prompt
    )

    # -------------------------------
    # QA prompt
    # -------------------------------
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say you don't know. "
        "Use four sentences maximum and keep the answer concise."
        "\n\n"
        "{context}"
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

    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    st.success("✅ PDFs processed. You can now ask questions.")

    user_input = st.text_input("Your question:")
    if user_input:
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        st.write("**Assistant:**", response["answer"])
else:
    st.info("Upload one or more PDFs to begin.")
