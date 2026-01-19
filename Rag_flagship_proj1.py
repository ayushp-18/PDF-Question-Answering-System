import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


st.set_page_config(page_title="PDF Q&A (OpenAI RAG)", layout="wide")
st.title("📄 PDF Question Answering System (OpenAI + RAG)")

# Check OpenAI key
if "OPENAI_API_KEY" not in st.secrets:
    st.error("OPENAI_API_KEY is missing in Streamlit secrets.")
    st.stop()

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# session
session_id = st.text_input("Session ID", value="default")

# store memory
if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

# Upload PDFs
uploaded_documents = st.file_uploader(
    "Upload PDF(s)",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_documents:
    documents = []

    for uploaded_document in uploaded_documents:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_document.getvalue())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        documents.extend(docs)

        try:
            os.remove(tmp_path)
        except:
            pass

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    # Vector store
    vector_store = FAISS.from_documents(chunks, embeddings)

    # ✅ SIMPLE retriever (stable)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # History aware question rewriting
    context_system_prompt = (
        "Given a chat history and the latest user question, rewrite it as a standalone question. "
        "Do NOT answer the question."
    )

    context_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", context_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_prompt)

    # Answering prompt
    qa_system_prompt = (
        "You are a PDF question answering assistant. "
        "Answer ONLY using the context below. "
        "If the answer is not in the context, say: 'I don't know based on the PDF.'\n\n"
        "Context:\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # Ask question
    question = st.text_input("Ask a question from the PDF:")

    if question:
        result = conversational_rag_chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}}
        )

        st.subheader("✅ Answer")
        st.write(result["answer"])

else:
    st.info("Upload PDFs to start asking questions.")
