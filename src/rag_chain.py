"""
Final Optimized RAG Chain.
Fixed retrieval issues by increasing k and optimizing prompt for attached numbers.
"""

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

def get_rag_chain():
    # 1. Load Embeddings
    embeddings = FastEmbedEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5")
    vectorstore = Chroma(
        persist_directory="data/chroma_db",
        embedding_function=embeddings,
        collection_name="rag_collection"
    )
    
    # 2. INCREASED K to 10: Ensures the specific chunk is captured despite spacing issues
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # 3. Initialize LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1,
        max_tokens=500
    )

     # 4. Persian-Optimized Prompt
    template = """You are an expert Persian Document Analyst AI.
Your task is to extract specific information from the provided Persian text.

Rules:
1. Answer ONLY based on the provided context.
2. IGNORE OCR/FONT TYPOS: The text may contain typos like "تسهیالت" (instead of تسهیلات) or "اطالعات" (instead of اطلاعات). Understand the true meaning.
3. FORM PLACEHOLDERS: Treat "......." or "..............." as indicators that the text immediately following them is the answer.
4. If the information is truly missing, strictly say: "در سند به وضوح ذکر نشده است".
5. Format your answer clearly and concisely in Persian.
6.If the question is about the general topic or summary of the document, infer it from the provided context.

Context: {context}

Question: {question}

Answer:"""
    
    prompt = PromptTemplate.from_template(template)

    # 5. Helper function
    def format_docs(docs):
        if not docs:
            return "No relevant context found."
        return "\n\n---\n\n".join([f"صفحه {doc.metadata.get('page', '?')}: {doc.page_content}" for doc in docs])

    # 6. Build the Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, vectorstore


if __name__ == "__main__":
    try:
        print("✅ Loading Optimized RAG Chain...")
        chain, vectorstore = get_rag_chain()
        print("✅ RAG Chain loaded successfully!")
        
        test_question = "موضوع اصلی سند چیست؟"
        print(f"\n🤖 Asking: {test_question}")
        
        response = chain.invoke(test_question)
        print(f"\n💡 FINAL ANSWER:\n{response}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()