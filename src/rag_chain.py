"""
RAG Chain module. Connects the Vector Store retriever with Groq LLM for ultra-fast answers.
"""

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()


def get_rag_chain():
    """
    Builds and returns the RAG chain for question answering using Groq.
    """
    # 1. Load the Vector Store and create a retriever
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="data/chroma_db",
        embedding_function=embeddings,
        collection_name="rag_collection"
    )
    
    # Retrieve top 3 most relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 2. Initialize the LLM with Groq
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1,
        max_tokens=500
    )

    # 3. Define a simpler, more robust Prompt Template
    template = """You are a helpful and precise AI assistant. 
Answer the question based ONLY on the following context. 
If the question is about the general topic or summary of the document, infer it from the provided context.
If the answer is absolutely not in the context, strictly say: "I don't know based on the provided document."

Context: {context}

Question: {question}

Answer:"""
    
    prompt = PromptTemplate.from_template(template)

    # 4. Helper function to format retrieved documents
    def format_docs(docs):
        if not docs:
            return "No relevant context found."
        return "\n\n".join(doc.page_content for doc in docs)

    # 5. Build the Runnable Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, Chroma(
        persist_directory="data/chroma_db",
        embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        collection_name="rag_collection"
    )


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING RAG CHAIN (Powered by Groq) - DEBUG MODE")
    print("=" * 60)
    
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Error: GROQ_API_KEY not found in .env file.")
    else:
        print("✅ GROQ_API_KEY loaded successfully.")
        print("🔄 Loading RAG Chain...")
        
        try:
            chain, vectorstore = get_rag_chain()
            print("✅ RAG Chain loaded successfully!")
            
            # 🔍 DEBUG: Let's see what's actually in the database
            print("\n🔍 DEBUG: Checking database content...")
            all_docs = vectorstore.similarity_search("a", k=1) # Dummy search to check if DB has data
            print(f"   Total chunks in DB: {len(vectorstore.get()['ids'])}")
            
            # 💡 IMPORTANT: Change this question to something SPECIFIC in your PDF!
            # Example: If your PDF is a resume, ask: "What is the person's name?" or "What are their skills?"
            test_question = "What is the main topic of this document?" 
            
            print(f"\n🤖 Asking: {test_question}")
            
            # Let's also see what the retriever finds for this specific question
            retrieved_docs = vectorstore.similarity_search(test_question, k=3)
            print(f"   🔍 Retrieved {len(retrieved_docs)} chunks for this question.")
            
            response = chain.invoke(test_question)
            print(f"\n💡 Answer:\n{response}")
            
        except Exception as e:
            print(f"\n❌ Error during RAG chain execution: {e}")