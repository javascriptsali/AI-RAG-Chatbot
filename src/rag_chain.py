"""
RAG Chain module. Optimized for Persian Document Extraction.
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
    # 1. Load the Vector Store
    embeddings = FastEmbedEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5")
    vectorstore = Chroma(
        persist_directory="data/chroma_db",
        embedding_function=embeddings,
        collection_name="rag_collection"
    )
    
    # 2. Initialize the LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1,
        max_tokens=500
    )

    # 3. Setup Retriever (Increased k to 5 to capture more context)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 4. Define the Prompt Template (Your excellent prompt)
    template = """You are an expert Persian Document Analyst AI.
Your task is to extract specific information from the provided Persian text (which may contain OCR errors or line breaks).

Rules:
1. Answer ONLY based on the provided context.
2. If a specific field (like area, address, or owner name) is not clearly mentioned or is garbled in the context, strictly say: "در سند به وضوح ذکر نشده است".
3. Do NOT guess, infer, or correct potential OCR typos. Report exactly what is in the text.
4. Format your answer clearly using bullet points.
5. If the question is about the general topic or summary of the document, infer it from the provided context.
6. If the information is split across lines (e.g., a label on one line and the value on the next), you MUST connect them logically to form a complete answer.
7. Do NOT guess or hallucinate numbers or addresses.

Context: {context}

Question: {question}

Answer:"""
    
    prompt = PromptTemplate.from_template(template)

    # 5. Helper function
    def format_docs(docs):
        if not docs:
            return "No relevant context found."
        return "\n\n".join(doc.page_content for doc in docs)

    # 6. Build the Runnable Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, vectorstore


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING RAG CHAIN (Powered by Groq)")
    print("=" * 60)
    
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Error: GROQ_API_KEY not found.")
    else:
        try:
            chain, vectorstore = get_rag_chain()
            print("✅ RAG Chain loaded successfully!")
            print(f"🔍 Total chunks in DB: {len(vectorstore.get()['ids'])}")
            
            test_question = "در دیالوگ ضحاک فونت باید چگونه باشد؟"
            print(f"\n🤖 Asking: {test_question}")
            
            response = chain.invoke(test_question)
            print(f"\n💡 Answer:\n{response}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")