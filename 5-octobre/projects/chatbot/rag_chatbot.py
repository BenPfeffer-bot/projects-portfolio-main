# src/rag_chatbot.py
import os
import glob
import logging
import sys
import pandas as pd
from typing import List
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

sys.path.append("/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/projects-portfolio-main/5-octobre")
from src.config import BASE_DIR, load_logger

logger = load_logger()

###############################################################################
# CONFIGURATION
###############################################################################

MODEL_NAME = "google/flan-t5-base"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ANALYSIS_DIR = os.path.join(BASE_DIR, "data", "analysis")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "data", "embeddings", "analysis_docs")
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

###############################################################################
# DOCUMENT LOADING & PROCESSING
###############################################################################


def load_analysis_documents(analysis_dir: str) -> List[Document]:
    """
    Load analysis CSV files from analysis_dir, convert them into text documents.
    We also add a dataset overview document for context.
    """
    documents = []

    dataset_description = """
    This dataset and analysis files contain metrics related to e-commerce performance:
    - Orders, customers, revenues, average order value (AOV), customer lifetime value (CLV)
    - Geographic distributions of sales
    - Payment methods usage and associated revenues
    - Monthly/Yearly trends, growth metrics, cancellations, refunds
    - Segmentation analyses (RFM, cohorts, new vs returning customers)
    The data has been processed and cleaned. The CSV files in 'analysis' directory represent summarized metrics or insights derived from raw cart and order data.
    """
    overview_doc = Document(page_content=dataset_description.strip(), metadata={"source": "dataset_overview"})
    documents.append(overview_doc)

    csv_files = glob.glob(os.path.join(analysis_dir, "*.csv"))
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        content = df.to_string(index=True)
        doc_name = os.path.basename(csv_file)
        doc = Document(page_content=content, metadata={"source": doc_name})
        documents.append(doc)

    return documents


def chunk_documents(documents: List[Document], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[Document]:
    """
    Chunk large documents into smaller pieces.
    """
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=chunk_size, chunk_overlap=overlap)
    chunked_docs = []
    for doc in documents:
        for chunk in text_splitter.split_text(doc.page_content):
            chunked_docs.append(Document(page_content=chunk, metadata=doc.metadata))
    return chunked_docs


###############################################################################
# MODEL & RETRIEVER SETUP
###############################################################################


def load_llm(model_name: str = MODEL_NAME) -> HuggingFacePipeline:
    """
    Load a local Hugging Face model for text generation.
    """
    logger.info(f"Loading LLM: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=1024, device="cpu")
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


def build_vectorstore(docs: List[Document], embedding_model: str = EMBEDDING_MODEL, persist_dir: str = VECTORSTORE_DIR):
    """
    Build or load a Chroma vector store from documents.
    """
    logger.info(f"Building vector store at {persist_dir} with embedding model {embedding_model}")
    embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
    vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_dir, collection_name="analysis_docs")
    vectorstore.persist()
    return vectorstore


###############################################################################
# PROMPT SETUP
###############################################################################

template = """You are a data analyst assistant for an e-commerce company.
You have access to analyzed metrics, summaries, and tables.
Your goal is to help answer any question regarding these datasets accurately.

Use the following context (extracted from internal analyses) to answer the question. If unsure, just say so.

{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

###############################################################################
# MAIN FUNCTION
###############################################################################


def main():
    if not os.path.exists(ANALYSIS_DIR):
        logger.error("Analysis directory not found. Run your analysis first.")
        return

    logger.info("Loading and preparing documents...")
    raw_docs = load_analysis_documents(ANALYSIS_DIR)
    docs = chunk_documents(raw_docs, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    # Build or load vectorstore
    if not os.path.exists(VECTORSTORE_DIR) or len(os.listdir(VECTORSTORE_DIR)) == 0:
        vectorstore = build_vectorstore(docs)
    else:
        logger.info(f"Loading existing vector store from {VECTORSTORE_DIR}")
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = Chroma(collection_name="analysis_docs", persist_directory=VECTORSTORE_DIR, embedding_function=embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = load_llm(MODEL_NAME)

    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff", chain_type_kwargs={"prompt": PROMPT}, return_source_documents=True)

    print("Reliable RAG-based chatbot assistant is ready. Type your question. Type 'exit' to quit.")
    while True:
        user_query = input("User: ")
        if user_query.strip().lower() in ["exit", "quit"]:
            break
        try:
            response = chain({"query": user_query})
            answer = response["result"]
            source_docs = response.get("source_documents", [])
            if source_docs:
                sources = [doc.metadata.get("source", "unknown") for doc in source_docs]
                unique_sources = list(set(sources))
                answer += f"\n\nSources: {', '.join(unique_sources)}"
            print("Assistant:", answer)
        except Exception as e:
            logger.error(f"Error during retrieval or generation: {e}", exc_info=True)
            print("Assistant: Sorry, I encountered an error. Please try again.")


if __name__ == "__main__":
    main()
