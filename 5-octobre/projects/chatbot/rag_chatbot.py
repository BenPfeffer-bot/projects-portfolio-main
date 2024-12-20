# src/rag_chatbot.py

import os
import glob
import logging
import sys
import json
import pandas as pd
from typing import List, Dict
from datetime import datetime
import dateparser
import spacy
import torch

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from spacy.matcher import Matcher

# Adjust the path to your environment
sys.path.append("/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/projects-portfolio-main/5-octobre")
from src.config import BASE_DIR, load_logger

logger = load_logger()

###############################################################################
# CONFIGURATION
###############################################################################

MODEL_NAME = "google/flan-t5-xxl"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ANALYSIS_DIR = os.path.join(BASE_DIR, "data", "analysis")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "data", "embeddings", "analysis_docs")
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

# Ensure spaCy language model is available
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
    nlp = None

###############################################################################
# USER PERSONA / ROLE
###############################################################################
user_persona = os.environ.get("USER_PERSONA", "GENERAL")

persona_instructions = ""
if user_persona == "MARKETING":
    persona_instructions = "The user is from the marketing team. Focus on metrics related to campaigns, conversion funnels, and acquisition costs."
elif user_persona == "FINANCE":
    persona_instructions = "The user is from the finance department. Emphasize revenue, costs, margins, and profitability metrics."
else:
    persona_instructions = "The user is looking for general insights on e-commerce performance."

###############################################################################
# DOMAIN KNOWLEDGE
###############################################################################
domain_instructions = """You are a data analyst assistant for an e-commerce company specializing in luxury jewelry.
Key metrics include:
- Revenue (monthly, yearly, YOY changes, forecasts)
- Orders (count, monthly trends, YOY)
- Unique Customers (new vs returning, monthly counts)
- Average Order Value (AOV)
- Customer Lifetime Value (CLV)
- Churn Rate (customer retention)
- Refunds, Cancellations
- Cohort retention, RFM segments
- CAC (Customer Acquisition Cost), LTV:CAC ratio"""

company_lexicon = """
KPI definitions:
- Total Orders: Count of orders in a period
- Total Revenue: Sum of 'Total' column in that period
- AOV: Total Revenue / Total Orders
- CLV: AOV * Purchase Frequency * Customer Lifetime
- Churn Rate: (Customers Lost / Customers at Start) * 100
- CAC: Marketing Spend / New Customers Acquired
"""

###############################################################################
# ADVANCED PROMPT WITH CHAIN-OF-THOUGHT
###############################################################################
template = """
Question: {question}

Context from documents:
{context}

Chat History:
{chat_history}

Instructions:
Based on the context and chat history, provide a clear and concise answer.
If specific data is not available in the context, indicate what information is missing.

Answer:"""

PROMPT = PromptTemplate(template=template, input_variables=["context", "question", "chat_history"])


###############################################################################
# DOCUMENT LOADING & PROCESSING
###############################################################################
def load_analysis_documents(analysis_dir: str) -> List[Document]:
    documents = []
    dataset_description = "This dataset and analysis files contain metrics related to e-commerce performance."
    overview_doc = Document(page_content=dataset_description.strip(), metadata={"source": "dataset_overview"})
    documents.append(overview_doc)

    csv_files = glob.glob(os.path.join(analysis_dir, "*.csv"))
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        content = df.to_string(index=True)
        doc_name = os.path.basename(csv_file)
        doc = Document(page_content=content, metadata={"source": doc_name})
        documents.append(doc)

    # If there's a textual summary file
    summary_file = os.path.join(analysis_dir, "analysis_summary.md")
    if os.path.exists(summary_file):
        with open(summary_file, "r") as f:
            summary_text = f.read()
        summary_doc = Document(page_content=summary_text, metadata={"source": "analysis_summary"})
        documents.append(summary_doc)

    return documents


def build_vectorstore(documents: List[Document], vectorstore_dir: str):
    if not os.path.exists(vectorstore_dir):
        os.makedirs(vectorstore_dir)

    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    # Split docs into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_docs = []
    split_metas = []
    for doc, meta in zip(texts, metadatas):
        chunks = text_splitter.split_text(doc)
        for chunk in chunks:
            split_docs.append(chunk)
            split_metas.append(meta)

    vectorstore = Chroma.from_texts(split_docs, embeddings, metadatas=split_metas, collection_name="analysis_collection", persist_directory=vectorstore_dir)
    vectorstore.persist()
    return vectorstore


###############################################################################
# NLU & DATE NORMALIZATION
###############################################################################
def interpret_query(query: str) -> str:
    """
    Use spaCy for entity extraction, dateparser for date normalization.
    Return a string describing what was found: entities, dates, etc.
    """
    if nlp is None:
        return "No NLU available."

    doc = nlp(query)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    # Attempt date normalization
    dates = [ent.text for ent, label in entities if label in ("DATE", "TIME")]
    normalized_dates = []
    for d in dates:
        parsed = dateparser.parse(d)
        if parsed:
            normalized_dates.append((d, parsed.isoformat()))

    interpretation = f"Entities found: {entities}. Normalized dates: {normalized_dates}."
    return interpretation


###############################################################################
# LLM & CHAIN SETUP
###############################################################################
def create_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    llm_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,  # Limit output length
        device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
    )
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    return llm


def create_qa_chain(vectorstore):
    """
    Create a conversational QA chain with proper memory handling
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Create memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

    # Initialize the chain with return_source_documents
    chain = ConversationalRetrievalChain.from_llm(
        llm=create_llm(),
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        chain_type="stuff",  # Use 'stuff' chain type for simpler document handling
        verbose=True,  # Add verbosity for debugging
    )
    return chain


###############################################################################
# FEEDBACK COLLECTION
###############################################################################
def save_feedback(user_question: str, answer: str, feedback: str, feedback_file="feedback.json"):
    """
    Save user feedback to a local JSON file. Append new feedback to existing file.
    """
    entry = {"timestamp": datetime.utcnow().isoformat(), "question": user_question, "answer": answer, "feedback": feedback}
    if os.path.exists(feedback_file):
        with open(feedback_file, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(entry)
    with open(feedback_file, "w") as f:
        json.dump(data, f, indent=2)


###############################################################################
# MAIN EXECUTION
###############################################################################
if __name__ == "__main__":
    # Load documents
    documents = load_analysis_documents(ANALYSIS_DIR)

    # Build or load vectorstore
    if os.path.exists(VECTORSTORE_DIR):
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embeddings)
    else:
        vectorstore = build_vectorstore(documents, VECTORSTORE_DIR)

    # Create QA chain
    qa_chain = create_qa_chain(vectorstore)

    # Example interaction loop
    print("Chatbot ready! Ask a question, or type 'exit' to quit.")

    while True:
        user_query = input("User: ")
        if user_query.lower().strip() == "exit":
            break

        try:
            # Run query through chain
            response = qa_chain({"question": user_query})

            if "answer" in response:
                answer = response["answer"]
                print("Assistant:", answer)

                # Ask for feedback
                feedback = input("Was this answer helpful? (y/n or comment): ")
                if feedback.strip():
                    save_feedback(user_query, answer, feedback.strip())
            else:
                print("Assistant: I apologize, but I couldn't generate a proper response. Please try rephrasing your question.")

        except Exception as e:
            print(f"Error processing query: {str(e)}")
            logger.error(f"Error in chat interaction: {str(e)}", exc_info=True)

    print("Goodbye!")
