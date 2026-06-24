import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

st.set_page_config(page_title="Zyro HR Help Desk", page_icon="💬")
st.title("💬 Zyro Dynamics HR Help Desk")

CORPUS_PATH = "data/"

@st.cache_resource
def load_pipeline():
    if not os.path.exists(CORPUS_PATH):
        st.error(f"Folder '{CORPUS_PATH}' not found. Make sure 'data/' is in your repo root, next to app.py.")
        st.stop()

    pdf_files = [f for f in os.listdir(CORPUS_PATH) if f.lower().endswith(".pdf")]
    if len(pdf_files) == 0:
        st.error(f"No PDF files found inside '{CORPUS_PATH}'. Upload the 11 HR policy PDFs there.")
        st.stop()

    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()

    if len(documents) == 0:
        st.error("PDFs were found but could not be loaded. Check the files aren't corrupted.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4})

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, max_tokens=512)

    rag_prompt = ChatPromptTemplate.from_template(
        "You are an HR assistant for Zyro Dynamics. Answer ONLY using the context below. "
        "If the answer isn't in the context, say you
