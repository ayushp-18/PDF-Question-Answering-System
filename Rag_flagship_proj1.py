#Importing neccesary libraries
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever,EnsembleRetriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import tabula
import os

#Setting up our env
# from dotenv import load_dotenv
# load_dotenv()

#Getting the Hugging Face Token and GroqApiKey
os.environ['HF_TOKEN']=st.secrets["HF_TOKEN"]
groq_api_key=st.secrets["GROQ_API_KEY"]
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#Setting up streamlit
st.title("Conversational Chat Bot with pdf upload support and chat history")
st.write("Upload pdf and chat with along the content of the PDF")
temperature=st.slider("Set your temperature as you require",0.0,1.0,0.7)

#Let set the groq api key before hand since its not deployment ready yet
llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama3-70b-8192",temperature=temperature)

#Now lets start with getting the session id 
session_id=st.text_input("Enter your session id",value='default_session')

#Stateful management of Chat History
if 'store' not in st.session_state:
    st.session_state.store={}

#Uploading the pdf
uploaded_documents=st.file_uploader("Choose A PDf file to upload",type="pdf",accept_multiple_files=True)
if uploaded_documents:
    documents=[]
    for uploaded_document in uploaded_documents:
        temp_pdf=f"./temp.pdf" 
        with open(temp_pdf,'wb') as file:
            file.write(uploaded_document.getvalue())
            file_name=uploaded_document.name
        loader=PyPDFLoader(temp_pdf)
        docs=loader.load()
        documents.extend(docs)


    #Creating the chunks of the document 
    # text_splitter1=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=20)
    # text_splitter2=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=75)
    text_splitter3=RecursiveCharacterTextSplitter(chunk_size=3800,chunk_overlap=1000)
    # chunks1=text_splitter1.split_documents(documents)
    # chunks2=text_splitter2.split_documents(documents)
    chunks3=text_splitter3.split_documents(documents) 

    #Storing final product of applying the embedder pattern to the chunks of documents
    #and storing it in vector db
    vector_store=FAISS.from_documents(documents=chunks3,embedding=embeddings)

    #Making the vector as a retreival class
    faiss_semantic_retriever=vector_store.as_retriever()
    bm25_retriever = BM25Retriever.from_documents(documents=chunks3)
    retriever = EnsembleRetriever(retrievers=[bm25_retriever,faiss_semantic_retriever],
                                       weights=[0.5,0.5])

    #This system prompt is designed such that we get a standalone question without question
    #being referenced in the past conversation
    context_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
    
    context_prompt=ChatPromptTemplate.from_messages(
                [
                    ("system", context_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

    #Creating a history aware retreiver
    #Combines both past information and context retreived information and fomulates it in one question
    #Output is a set of documents
    history_aware_retriever=create_history_aware_retriever(llm,retriever,context_prompt)

    #Question answering prompt creating task starts here
    system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use four sentences maximum and keep the "
                "answer concise."
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
    
    #Create a chain for passing a list of Documents to a model.
    question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
    #Binding 2 chains together
    rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

    #Creating storgae functionality
    def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]

    conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

    user_input = st.text_input("Your question:")
    if user_input:
            session_history=get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id":session_id}
                },  # constructs a key "abc123" in `store`.
            )
            # st.write(st.session_state.store)
            st.write("Assistant:", response['answer'])
            # st.write("Chat History:", session_history.messages)
