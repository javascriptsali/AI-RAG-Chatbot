"""
Data ingestion module for the RAG Chatbot.
Optimized for Scanned Persian PDFs using High-DPI OCR.
"""

import os
import gc
import numpy as np
import fitz         # PyMuPDF
import easyocr
import cv2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document

# Initialize OCR reader for Persian and English
# Note: First run downloads models (~50MB)
ocr_reader = easyocr.Reader(['fa', 'en'], gpu=False)


def extract_text_with_high_dpi_ocr(pdf_path: str) -> list:
    documents = []
    doc = fitz.open(pdf_path)
    
    print("🔍 Running High-DPI OCR with Image Pre-processing...")
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # 1. High DPI Zoom
        zoom = 2.5  # افزایش به 2.5 برای وضوح بیشتر حروف فارسی
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # 2. Convert to OpenCV format
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        # 3. Image Pre-processing for better OCR (Grayscale + Thresholding)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Adaptive thresholding makes text stand out from noisy backgrounds
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        
        # 4. Run OCR on the cleaned image
        # detail=0 returns only text, which is cleaner for RAG
        result = ocr_reader.readtext(thresh, detail=0, paragraph=True)
        
        page_text = " ".join(result)
        
        if page_text.strip():
            documents.append(Document(
                page_content=page_text,
                metadata={"page": page_num + 1, "source": os.path.basename(pdf_path)}
            ))
            
    doc.close()
    return documents


def ingest_pdf(pdf_path: str, persist_directory: str = "data/chroma_db") -> str:
    """
    Processes a PDF file, forcing OCR for reliable Persian text extraction.
    """
    print(f"📥 [Step 1] Processing PDF: {pdf_path}")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
    
    # FORCE OCR for this pipeline to guarantee clean Persian text
    raw_documents = extract_text_with_high_dpi_ocr(pdf_path)
    total_pages = len(raw_documents)
    
    if total_pages == 0:
        raise ValueError("No text could be extracted from the PDF. Please check the file.")
        
    print(f"✅ Successfully extracted text from {total_pages} pages.")
    
    # --- Page limit check ---
    MAX_PAGES = 200
    if total_pages > MAX_PAGES:
        raise ValueError(f"Document too large! Maximum {MAX_PAGES} pages allowed.")
    
    # 2. Split text into chunks
    print("\n✂️ [Step 2] Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,      # افزایش سایز برای در بر گرفتن جملات طولانی‌تر
        chunk_overlap=400,    # افزایش همپوشانی برای اطمینان از اینکه "کلید" و "مقدار" از هم جدا نمی‌شوند
        length_function=len,
        separators=["\n\n", "\n", " ", ""] # اولویت شکستن از پاراگراف، سپس خط، سپس کلمه
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"✅ Created {len(chunks)} text chunks.")
    
    # Free memory
    del raw_documents
    gc.collect()
    
    # 3. Initialize Embeddings (Multilingual)
    print("\n🧠 [Step 3] Initializing Embedding Model...")
    embeddings = FastEmbedEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5")
    
    # 4. Store in Vector Database using BATCH PROCESSING
    print("\n💾 [Step 4] Storing embeddings in ChromaDB (Batch Processing)...")
    os.makedirs(persist_directory, exist_ok=True)
    
    if os.path.exists(os.path.join(persist_directory, "chroma.sqlite3")):
        print("🗑️ Existing database found. Clearing old data...")
        import shutil
        shutil.rmtree(persist_directory)
        os.makedirs(persist_directory, exist_ok=True)
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="rag_collection"
    )
    
    # Batch Processing to prevent OOM
    BATCH_SIZE = 50
    total_chunks = len(chunks)
    
    for i in range(0, total_chunks, BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        vectorstore.add_documents(documents=batch)
        print(f"   Processed batch {i // BATCH_SIZE + 1}: {min(i + BATCH_SIZE, total_chunks)}/{total_chunks} chunks")
        gc.collect()
    
    print(f"\n✅ Successfully ingested data into: {persist_directory}")
    return f"Success: Processed {len(chunks)} chunks from {os.path.basename(pdf_path)} ({total_pages} pages)"


if __name__ == "__main__":
    sample_pdf_path = "data/uploads/sample.pdf"
    
    print("=" * 60)
    print("RAG INGESTION PIPELINE TEST (High-DPI OCR Forced)")
    print("=" * 60)
    
    try:
        result = ingest_pdf(sample_pdf_path)
        print(f"\n🎉 {result}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()