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
    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4})

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, max_tokens=512)

    rag_prompt = ChatPromptTemplate.from_template(
        "You are an HR assistant for Zyro Dynamics. Answer ONLY using the context below. "
        "If the answer isn't in the context, say you don't have that information.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    oos_prompt = ChatPromptTemplate.from_template(
        "Is the following question related to HR policies (leave, WFH, conduct, "
        "performance, compensation, IT security, POSH, onboarding, travel)? "
        "Answer only YES or NO.\n\nQuestion: {question}"
    )
    return retriever, llm, rag_prompt, oos_prompt

retriever, llm, rag_prompt, oos_prompt = load_pipeline()

def ask_bot(question):
    check_chain = oos_prompt | llm | StrOutputParser()
    verdict = check_chain.invoke({"question": question}).strip().upper()
    if "NO" in verdict:
        return "I can only answer HR-related questions from Zyro Dynamics policy documents.", []

    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    chain = rag_prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return answer, docs

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask an HR question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    answer, sources = ask_bot(prompt)

    with st.chat_message("assistant"):
        st.write(answer)
        if sources:
            with st.expander("Sources"):
                for i, doc in enumerate(sources, 1):
                    page = doc.metadata.get("page", "?")
                    source_file = doc.metadata.get("source", "unknown")
                    st.write(f"**{i}.** {source_file} (page {page})")
                    st.caption(doc.page_content[:200] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})
