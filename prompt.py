import streamlit as st
import tempfile
from langchain_community.document_loaders  import PyPDFLoader
#from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
#from langchain.memory import ConversationalBufferMemory
from transformers import pipeline

#memory = ConversationalBufferMemory()

def load_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    return documents

def main():
    st.title("PDF Chatbot with Hugging Face LLM")

    pdf_file = st.file_uploader("Upload your PDF", type=["pdf"])

    if pdf_file is not None:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 400 , chunk_overlap=50)
        docs= load_pdf(pdf_file)
        documents = text_splitter.split_documents(docs)
        st.success(f"Loaded {len(documents)} document chunks")

        # Initialize embeddings model
        embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

        # Create vectorstore (Chroma) from documents
        vectordb = Chroma.from_documents(documents, embeddings)

        # Setup retriever
        retriever = vectordb.as_retriever(search_kwargs={"k": 5})

        # Load HF pipeline (e.g., a small text-generation model)
        hf_model_name = "google/flan-t5-large"
        pipe = pipeline("text2text-generation", model=hf_model_name, max_length=1024)

        # Wrap HF pipeline as LangChain LLM
        llm = HuggingFacePipeline(pipeline=pipe)

        # Setup RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)#memory = memory)

        # Chat interface
        query = st.text_input("Ask a question about your PDF")

        if query:
            answer = qa_chain.run(query)
            st.write("**Answer:**")
            st.write(answer)

if __name__ == "__main__":
    main()

#other CODE