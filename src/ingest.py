"""
Data ingestion module for the RAG Chatbot.
Implements batch processing to prevent Out-Of-Memory (OOM) crashes on large documents.
"""

import os
import gc
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings


def ingest_pdf(pdf_path: str, persist_directory: str = "data/chroma_db") -> str:
    """
    Processes a PDF file using batch processing to avoid memory issues.
    """
    print(f"📥 [Step 1] Loading PDF from: {pdf_path}")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
    
    # 1. Load the PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    total_pages = len(documents)
    print(f"✅ Loaded {total_pages} pages.")
    
    # --- NEW: Page limit check ---
    MAX_PAGES = 200
    if total_pages > MAX_PAGES:
        raise ValueError(f"Document too large! Maximum {MAX_PAGES} pages allowed. Your document has {total_pages} pages.")
    # -----------------------------
    
    # 2. Split text into chunks
    print("\n✂️ [Step 2] Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} text chunks.")
    
    # Free memory from original documents
    del documents
    gc.collect()
    
    # 3. Initialize Embeddings
    print("\n🧠 [Step 3] Initializing Embedding Model...")
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    # 4. Store in Vector Database using BATCH PROCESSING
    print("\n💾 [Step 4] Storing embeddings in ChromaDB (Batch Processing)...")
    os.makedirs(persist_directory, exist_ok=True)
    
    # Check if collection already exists, if so, delete it to avoid duplicates
    if os.path.exists(os.path.join(persist_directory, "chroma.sqlite3")):
        print("️ Existing database found. Clearing old data...")
        import shutil
        shutil.rmtree(persist_directory)
        os.makedirs(persist_directory, exist_ok=True)
    
    # Initialize empty ChromaDB
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="rag_collection"
    )
    
    # --- NEW: Batch Processing to prevent OOM ---
    BATCH_SIZE = 50  # Process 50 chunks at a time
    total_chunks = len(chunks)
    
    for i in range(0, total_chunks, BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        vectorstore.add_documents(documents=batch)
        print(f"   Processed batch {i // BATCH_SIZE + 1}: {min(i + BATCH_SIZE, total_chunks)}/{total_chunks} chunks")
        
        # Free memory after each batch
        gc.collect()
    # --------------------------------------------
    
    print(f"\n✅ Successfully ingested data into: {persist_directory}")
    return f"Success: Processed {len(chunks)} chunks from {os.path.basename(pdf_path)} ({total_pages} pages)"


if __name__ == "__main__":
    sample_pdf_path = "data/uploads/sample.pdf"
    
    print("=" * 60)
    print("RAG INGESTION PIPELINE TEST (Batch Processing)")
    print("=" * 60)
    
    try:
        result = ingest_pdf(sample_pdf_path)
        print(f"\n🎉 {result}")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
    except ValueError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")