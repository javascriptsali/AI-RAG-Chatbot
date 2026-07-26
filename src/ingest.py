"""
Ultra-fast ingestion with SMALL BATCH processing to prevent RAM overload.
"""

import os
import gc
import fitz

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document


def extract_text_simple(pdf_path: str) -> list:
    documents = []
    doc = fitz.open(pdf_path)
    print("🔍 Extracting text from PDF...")
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text().strip()
        
        if text:
            documents.append(Document(
                page_content=text,
                metadata={"page": page_num + 1, "source": os.path.basename(pdf_path)}
            ))
    
    doc.close()
    return documents


def ingest_pdf(pdf_path: str, persist_directory: str = "data/chroma_db") -> str:
    print(f"📥 [Step 1] Processing PDF: {pdf_path}")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
    
    # 1. Extract text
    raw_documents = extract_text_simple(pdf_path)
    total_pages = len(raw_documents)
    print(f"✅ Extracted {total_pages} pages.")
    
    # 2. Chunking with LARGER chunk size (fewer chunks = less RAM)
    print("\n✂️ [Step 2] Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,      # Increased from 1500 to 2000
        chunk_overlap=300,    # Reduced from 400 to 300
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"✅ Created {len(chunks)} chunks.")
    
    del raw_documents
    gc.collect()
    
    # 3. Initialize Embeddings
    print("\n🧠 [Step 3] Initializing Embedding Model...")
    embeddings = FastEmbedEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5")
    
    # 4. Store in ChromaDB with VERY SMALL BATCHES
    print("\n💾 [Step 4] Storing embeddings in ChromaDB (Small Batches)...")
    
    # Delete old database
    if os.path.exists(persist_directory):
        print("🗑️ Clearing old database...")
        import shutil
        try:
            shutil.rmtree(persist_directory)
        except:
            pass
    os.makedirs(persist_directory, exist_ok=True)
    
    # Initialize empty ChromaDB
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="rag_collection"
    )
    
    # CRITICAL: Process in VERY SMALL batches (10 chunks at a time)
    BATCH_SIZE = 5  # Reduced from 10 to 5
    total_chunks = len(chunks)
    
    print(f"   Processing {total_chunks} chunks in batches of {BATCH_SIZE}...")
    
    for i in range(0, total_chunks, BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        vectorstore.add_documents(documents=batch)
        
        # Show progress
        progress = min(i + BATCH_SIZE, total_chunks)
        print(f"   ✓ Processed {progress}/{total_chunks} chunks")
        
        # CRITICAL: Free RAM after EACH batch
        gc.collect()
    
    print(f"\n✅ Successfully ingested data!")
    return f"Success: Processed {len(chunks)} chunks from {total_pages} pages"


if __name__ == "__main__":
    try:
        result = ingest_pdf("data/uploads/sample.pdf")
        print(f"\n🎉 {result}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()