"""
Streamlit Web Application for the RAG Chatbot.
Allows users to upload a PDF, process it, and chat with the document.
Optimized with caching for ultra-fast performance.
"""

import os
import sys
import streamlit as st
from dotenv import load_dotenv

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

from ingest import ingest_pdf  # type: ignore
from rag_chain import get_rag_chain  # type: ignore

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .chat-message { padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
    .user-message { background-color: #e3f2fd; border-left: 4px solid #2196f3; }
    .ai-message { background-color: #f1f8e9; border-left: 4px solid #4caf50; }
    </style>
""", unsafe_allow_html=True)


# --- CACHING THE RAG CHAIN FOR ULTRA-FAST PERFORMANCE ---
@st.cache_resource(show_spinner=False)
def load_rag_chain():
    """
    Loads and caches the RAG chain. 
    This runs only ONCE per session, making subsequent questions instant.
    """
    return get_rag_chain()
# --------------------------------------------------------


def main():
    st.title("🤖 AI Document Chatbot (RAG)")
    st.markdown("Upload a PDF document and ask questions about its contents. The AI will answer based **only** on the provided document.")

    # Sidebar for file upload and processing
    with st.sidebar:
        st.header(" Document Management")
        
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
        
        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            upload_dir = "data/uploads"
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, uploaded_file.name)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.info(f"Saved to: `{file_path}`")
            
            if st.button("🚀 Process Document & Build Vector DB", type="primary"):
                with st.spinner("Reading PDF, chunking text, and generating embeddings..."):
                    try:
                        result = ingest_pdf(file_path)
                        st.success(result)
                        st.session_state["document_processed"] = True
                        st.session_state["document_name"] = uploaded_file.name
                    except Exception as e:
                        st.error(f" Error processing document: {e}")
        else:
            st.warning("⚠️ Please upload a PDF file to begin.")

    # Main Chat Interface
    st.markdown("---")
    st.subheader("💬 Chat with your Document")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"].replace('\\', '\\\\'))

    if not st.session_state.get("document_processed", False):
        st.info("👈 Please upload and process a document from the sidebar first.")
    else:
        st.success(f"✅ Ready to chat with: **{st.session_state.get('document_name')}**")

        if prompt := st.chat_input("Ask a question about the document..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt.replace('\\', '\\\\'))

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        #  USE THE CACHED CHAIN - This makes it instant!
                        chain, _ = load_rag_chain()
                        response = chain.invoke(prompt)
                        
                        st.markdown(response.replace('\\', '\\\\'))
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                    except Exception as e:
                        error_msg = f"❌ An error occurred: {e}"
                        st.markdown(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "<p>Built with ❤️ using LangChain, ChromaDB, Groq, and Streamlit</p>"
        "<p>Part of AI Engineering Portfolio Project</p>"
        "</div>", 
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()