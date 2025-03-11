
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Streamlit UI
st.title("RAG Document Q&A with GROQ and Llama/Gemma")

# User input for API key
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = ""

st.session_state.groq_api_key = st.text_input(
    "Enter your Groq API Key:", 
    value=st.session_state.groq_api_key, 
    type="password"
)

# Ensure the user enters an API key
if not st.session_state.groq_api_key:
    st.warning("Please enter your Groq API key to proceed.")
    st.stop()

# Initialize LLM with user-provided API key
llm = ChatGroq(model_name="gemma2-9b-it", api_key=st.session_state.groq_api_key)

# Define prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

def create_vector_embedding():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    loader = PyPDFDirectoryLoader("research_papers")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    final_documents = text_splitter.split_documents(docs[:10])
    vectors = FAISS.from_documents(final_documents, embeddings)
    return vectors

user_prompt = st.text_input("Enter your query from the research paper")

if st.button("Generate Document Embeddings"):
    vectors = create_vector_embedding()
    st.session_state["vectors"] = vectors
    st.success("Vector database is ready!")

if user_prompt and "vectors" in st.session_state:
    retriever = st.session_state["vectors"].as_retriever()
    documents_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, documents_chain)
    response = retrieval_chain.invoke({"input": user_prompt})

    st.write("### Response:")
    st.write(response["answer"])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response.get("context", [])):
            st.write(doc.page_content)
            st.write("---")

