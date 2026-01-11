# PDF-Question-Answering-System


This project is a conversational chatbot that allows users to upload PDF documents and ask questions based on their content.  
It uses a Retrieval-Augmented Generation (RAG) approach with chat history awareness to generate accurate and context-aware responses.

---

## Try It Out

You can access the deployed Streamlit application here:

https://your-streamlit-app-url

---

## Project Overview

The application enables users to interact with PDF documents through natural language queries.  
It combines semantic search and keyword-based retrieval with a large language model to provide concise answers while maintaining conversational context across multiple turns.

---

## Key Features

- Upload and process multiple PDF files  
- Context-aware conversational question answering  
- Chat history management using session IDs  
- Hybrid document retrieval using FAISS and BM25  
- Adjustable response creativity using temperature control  
- Web-based interface built with Streamlit  

---

## Technology Stack

- Frontend: Streamlit  
- Language Model: Groq (LLaMA 3 – 70B)  
- Framework: LangChain  
- Embeddings: HuggingFace (all-MiniLM-L6-v2)  
- Vector Store: FAISS  
- Document Loader: PyPDFLoader  

---

## Project Structure

