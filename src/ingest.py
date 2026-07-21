"""
Data ingestion module for the RAG Chatbot.
Handles loading PDFs, chunking text, generating embeddings, and storing in ChromaDB.
Optimized for Streamlit Cloud deployment using fastembed.
"""

import os
import gc
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings


def ingest_pdf(pdf_path: str, persist_directory: str = "data/chroma_db") -> str:
    """
    Processes a PDF file and stores its embeddings in a local ChromaDB vector store.
    
    Args:
        pdf_path: Path to the PDF file to be processed.
        persist_directory: Directory where the ChromaDB will be saved.
        
    Returns:
        A success message with the number of chunks created.
    """
    print(f"📥 [Step 1] Loading PDF from: {pdf_path}")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
    
    # 1. Load the PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages.")
    
    # 2. Split text into chunks
    print("\n✂️ [Step 2] Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Optimized for large documents
        chunk_overlap=200,    # Maintains context between chunks
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} text chunks.")
    
    # 3. Initialize Embeddings using FastEmbed (lightweight, no PyTorch needed)
    print("\n🧠 [Step 3] Initializing Embedding Model (fastembed)...")
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    # 4. Store in Vector Database
    print("\n💾 [Step 4] Storing embeddings in ChromaDB...")
    os.makedirs(persist_directory, exist_ok=True)
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="rag_collection"
    )
    
    print(f"✅ Successfully ingested data into: {persist_directory}")
    # --- NEW: Force Garbage Collection to free up RAM ---
    print("🧹 Cleaning up memory...")
    gc.collect()
    # ----------------------------------------------------
    return f"Success: Processed {len(chunks)} chunks from {os.path.basename(pdf_path)}"


if __name__ == "__main__":
    sample_pdf_path = "data/uploads/sample.pdf"
    
    print("=" * 60)
    print("RAG INGESTION PIPELINE TEST")
    print("=" * 60)
    
    try:
        result = ingest_pdf(sample_pdf_path)
        print(f"\n🎉 {result}")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please place a sample PDF file named 'sample.pdf' in the 'data/uploads/' folder and try again.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")