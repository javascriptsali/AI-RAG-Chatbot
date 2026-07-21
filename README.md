# 🤖 AI Document Chatbot (RAG System)

A professional end-to-end Retrieval-Augmented Generation (RAG) application that allows users to upload any PDF document and ask questions about its contents. The AI answers strictly based on the provided document, eliminating hallucinations.

## 🌟 Live Demo

[Live Demo](https://ai-rag-chatbot-xdqy3xmreuckd5rbuexjhl.streamlit.app/)

## 📋 Project Overview

This project demonstrates a production-ready RAG pipeline, integrating vector databases, embedding models, and large language models (LLMs) to build an intelligent document question-answering system.

### Key Features

- ✅ **PDF Ingestion Pipeline**: Extracts, chunks, and embeds text from any PDF document
- ✅ **Vector Database**: Uses ChromaDB for efficient semantic search and retrieval
- ✅ **Ultra-Fast Inference**: Powered by Groq API (Llama 3.1) for sub-second response times
- ✅ **Local Embeddings**: Uses HuggingFace `all-MiniLM-L6-v2` for secure, cost-free vectorization
- ✅ **Interactive Web App**: Built with Streamlit for a clean, real-time chat interface
- ✅ **Optimized Performance**: Implements caching for instant responses on subsequent queries

## Project Structure

```tree
AI-RAG-Chatbot/
├── app/
│   └── streamlit_app.py        # Interactive web application
├── data/
│   ├── uploads/                # Uploaded PDF files
│   └── chroma_db/              # Local vector database (gitignored)
├── docs/
│   └── architecture.png        # System architecture diagram
├── src/
│   ├── __init__.py
│   ├── ingest.py               # PDF loading, chunking, and embedding pipeline
│   ├── rag_chain.py            # RAG chain logic (Retriever + LLM)
│   └── chat.py                 # Chat interface logic
├── app/
│   └── streamlit_app.py        # Interactive web application
├── docs/
│   └── architecture.png        # System architecture diagram
── .env                        # Environment variables (API keys)
├── .gitignore
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Quick Start

### Prerequisites

- Python 3.10+
- pip (Python package manager)
- A free Groq API key
- A free Hugging Face token

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/javascriptsali/AI-RAG-Chatbot.git
   cd AI-RAG-Chatbot
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   - **On Windows:**
   venv\Scripts\activate
   - **On Linux/Mac:**
   source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the project root and add your API keys:

   ```bash
   GROQ_API_KEY=gsk_your_groq_key_here
   HF_TOKEN=hf_your_huggingface_token_here
   ```

5. **Run the web application:**

   ```bash
   streamlit run app/streamlit_app.py
   ```

6. **Open your browser:**

   Visit `http://localhost:8501`

## 🧠 How It Works (The RAG Pipeline)

- Ingestion: The uploaded PDF is read using PyPDFLoader, split into chunks of 1000 characters with 200-character overlap using RecursiveCharacterTextSplitter.
- Embedding: Each chunk is converted into a 384-dimensional vector using the local all-MiniLM-L6-v2 model from HuggingFace.
- Storage: Vectors are stored in ChromaDB for fast semantic similarity search.
- Retrieval: When a user asks a question, the top 3 most relevant chunks are retrieved from the database.
- Generation: The retrieved context and the user's question are sent to Llama 3.1 via Groq API, which generates a precise answer based ONLY on the provided context.

## 🔧 Technical Stack

- Programming Language: Python 3.10+
- LLM Orchestration: LangChain & LangChain Community
- Vector Database: ChromaDB
- Embeddings: HuggingFace sentence-transformers
- LLM Provider: Groq API (Llama 3.1 8B Instant)
- PDF Processing: PyPDF
- Web Framework: Streamlit
- Version Control: Git & GitHub

## 💡 Challenges Solved

- Escape Character Rendering: Fixed Markdown rendering issues for backslashes in technical responses.
- General Question Handling: Engineered prompts to allow the model to infer general document topics from retrieved chunks.
- Performance Optimization: Implemented @st.cache_resource to cache the embedding model and RAG chain, reducing response. time from seconds to milliseconds.
- Context Window Management: Optimized chunk size (1000 chars) and overlap (200 chars) for large documents like books.

### ⚠️ System Limitations (Cloud Deployment)

- **Maximum File Size:** Optimized for PDF documents up to **15 MB** (approx. 50-100 pages) due to RAM constraints on the free tier of Streamlit Cloud.
- **Processing Time:** Larger documents may take 1-2 minutes to process and embed. Please be patient during the "Building Vector DB" phase.
- *For production use with large-scale documents, a dedicated server with 4GB+ RAM and batch-processing pipelines is recommended.*

## 🔮 Future Improvements

- [ ] Deploy on Streamlit Cloud for public access
- [ ] Add support for multiple document formats (DOCX, TXT, HTML)
- [ ] Implement conversation memory for multi-turn dialogues
- [ ] Add source citation (show which page the answer came from)
- [ ] Integrate with cloud storage (AWS S3, Google Drive)
- [ ] Add user authentication and document management

## 👨‍💻 Author

**Saleh Bakhtiyari**  
[LinkedIn Profile](https://linkedin.com/in/deve-loper-4870b5376)  
[GitHub Profile](https://github.com/javascriptsali)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

**If you found this project helpful, please give it a ⭐ on GitHub!**
