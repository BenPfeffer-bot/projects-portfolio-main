# src/rag_chatbot.py

import os
import glob
import logging
import sys
import pandas as pd
from typing import List, Dict
from datetime import datetime
import dateparser
import spacy

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from spacy.matcher import Matcher

sys.path.append("/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/projects-portfolio-main/5-octobre")
from src.config import BASE_DIR, load_logger

logger = load_logger()

###############################################################################
# CONFIGURATION
###############################################################################

# Switch to a more capable model (adjust as needed)
MODEL_NAME = "google/flan-t5-xxl"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ANALYSIS_DIR = os.path.join(BASE_DIR, "data", "analysis")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "data", "embeddings", "analysis_docs")
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

# Load spaCy model for NLU
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("spaCy model not found. Please run: python -m spacy download en_core_web_sm")
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
# DOMAIN INSTRUCTIONS & LEXICON
###############################################################################
domain_instructions = f"""
You are a data analyst assistant for an e-commerce company. The company sells luxury jewelry and home decor.
The key metrics include:
- Revenue (total monthly revenue, YOY changes, forecasting)
- Orders (total orders, monthly orders, YOY changes)
- Unique Customers (monthly unique customers, new vs returning)
- Average Order Value (AOV)
- Customer Lifetime Value (CLV)
- Churn Rate
- Refunds, Cancellations
- Cohort retention, RFM analysis
- CAC (Customer Acquisition Cost), LTV:CAC ratio

{persona_instructions}

They also analyze seasonal trends (Q1, Q2, holiday seasons), and promotions, and they sell product categories such as "Chairs", "Tables", "Sofas", "Beds", "Lighting".

The company uses the following terms frequently:
- "Monthly Revenue": total revenue aggregated by month.
- "YOY growth": year-over-year growth, comparing metrics from the same month last year.
- "Churn": percentage of customers who did not return period-to-period.
- "High-value customers": top 20% by spend.
- "CAC": average cost to acquire a new customer.

Always provide answers grounded in provided data and definitions. If unsure, say so.
When reasoning, consider best steps. The user query may contain relative time frames like "last month", "this quarter", or "the same month last year".
"""

company_lexicon = """
KPI definitions:
- Total Orders: Count of orders placed in a period.
- Total Revenue: Sum of 'Total' column of orders in that period.
- AOV: Total Revenue / Total Orders.
- CLV: AOV * Purchase Frequency * Customer Lifetime
- Churn Rate: (Customers Lost / Customers at Start) * 100
- CAC: Marketing Spend / New Customers Acquired
"""

###############################################################################
# ADVANCED PROMPT WITH CHAIN-OF-THOUGHT
###############################################################################
# We add hidden reasoning instructions:
template = """{domain_instructions}

{company_lexicon}

# Instructions:
You are a helpful assistant with access to analyzed metrics and documents. First, think step-by-step internally about what the user is asking and what data you have. Then answer.

**Chain of Thought (hidden reasoning)**
- Step 1: Parse user question, identify metrics, time periods, and comparisons.
- Step 2: Retrieve relevant context from provided data (context).
- Step 3: Use definitions and known KPIs to produce an accurate and concise answer.
- If uncertain, explain what additional info is needed.

**End of hidden reasoning**

Context:
{context}

NLU Interpretation:
{nlu_interpretation}

Question: {question}

Answer (visible to user):
"""

PROMPT = PromptTemplate(template=template, input_variables=["context", "question", "domain_instructions", "company_lexicon", "nlu_interpretation"])


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

    return documents


def chunk_documents(documents: List[Document], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[Document]:
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
    logger.info(f"Loading LLM: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=2048, device="cpu")
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


def build_vectorstore(docs: List[Document], embedding_model: str = EMBEDDING_MODEL, persist_dir: str = VECTORSTORE_DIR):
    logger.info(f"Building vector store at {persist_dir}")
    embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
    vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_dir, collection_name="analysis_docs")
    vectorstore.persist()
    return vectorstore


###############################################################################
# ADVANCED NLU FUNCTIONS
###############################################################################
def normalize_relative_dates(query: str) -> Dict[str, str]:
    """
    Interpret relative time expressions into concrete date ranges or normalized strings.
    Examples:
    - "last month" -> previous calendar month start and end
    - "same month last year" -> corresponding month last year
    - "last quarter" -> previous quarter
    - "Q1 2023" -> specific quarter start and end

    For simplicity, we'll return a dict with recognized times and their normalized forms.
    More complex logic might require a calendar library or domain-specific rules.
    """
    # Heuristics for common relative terms:
    recognized_times = {}
    now = datetime.now()

    if "last month" in query.lower():
        # Last month: the month before current month
        last_month = (now.replace(day=1) - pd.DateOffset(months=1)).to_pydatetime()
        start = last_month.replace(day=1)
        end = (start + pd.DateOffset(months=1) - pd.DateOffset(days=1)).to_pydatetime()
        recognized_times["last month"] = f"{start.date()} to {end.date()}"

    if "this month" in query.lower():
        start = now.replace(day=1)
        end = (start + pd.DateOffset(months=1) - pd.DateOffset(days=1)).to_pydatetime()
        recognized_times["this month"] = f"{start.date()} to {end.date()}"

    if "last year" in query.lower():
        # Interpret last year as entire previous year
        prev_year = now.year - 1
        recognized_times["last year"] = f"{prev_year}-01-01 to {prev_year}-12-31"

    # Similar logic could be added for quarters, weeks, YOY comparisons, etc.
    # For demonstration, we just handle a few cases.

    return recognized_times


def extract_domain_entities(query: str) -> Dict[str, List[str]]:
    """
    Extract domain-specific entities such as:
    - Metrics (e.g., revenue, orders, customers, AOV, CLV)
    - Products (Chairs, Tables, Sofas, Beds, Lighting)
    - Time expressions (handled partly by dateparser and custom logic)
    """
    if not nlp:
        return {"metrics": [], "products": [], "time_expressions": []}

    doc = nlp(query)
    metrics = []
    products = []
    time_expressions = []

    metric_keywords = ["revenue", "orders", "customers", "aov", "average order value", "clv", "churn", "growth", "refund", "cancellation", "cac"]
    product_categories = ["chairs", "tables", "sofas", "beds", "lighting"]

    # Detect metrics
    for token in doc:
        if token.text.lower() in metric_keywords and token.text.lower() not in metrics:
            metrics.append(token.text.lower())

    # Detect products
    for token in doc:
        if token.text.lower() in product_categories and token.text.lower() not in products:
            products.append(token.text.lower())

    # Extract DATE entities using spaCy
    for ent in doc.ents:
        if ent.label_ == "DATE":
            parsed_date = dateparser.parse(ent.text)
            if parsed_date:
                time_expressions.append(f"{ent.text} -> {parsed_date.date()}")
            else:
                time_expressions.append(ent.text)

    # Combine with relative date normalization
    relative_times = normalize_relative_dates(query)
    for k, v in relative_times.items():
        time_expressions.append(f"{k} -> {v}")

    return {"metrics": metrics, "products": products, "time_expressions": time_expressions}


def parse_query_nlu(query: str) -> str:
    """
    Parse the user query using NLU steps:
    - Entity extraction
    - Date normalization
    - Comparison detection
    Return a text summary of the interpretation.
    """
    entities = extract_domain_entities(query)

    comparison = False
    if "compare" in query.lower() or "versus" in query.lower() or "vs." in query.lower():
        comparison = True

    nlu_interpretation = (
        f"Metrics identified: {entities['metrics']}\n"
        f"Products identified: {entities['products']}\n"
        f"Time expressions identified: {entities['time_expressions']}\n"
        f"Comparison requested: {comparison}\n"
    )

    return nlu_interpretation.strip()


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

    if not os.path.exists(VECTORSTORE_DIR) or len(os.listdir(VECTORSTORE_DIR)) == 0:
        vectorstore = build_vectorstore(docs)
    else:
        logger.info(f"Loading existing vector store from {VECTORSTORE_DIR}")
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = Chroma(collection_name="analysis_docs", persist_directory=VECTORSTORE_DIR, embedding_function=embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = load_llm(MODEL_NAME)

    # Use conversational memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory, condense_question_prompt=PROMPT.partial(domain_instructions=domain_instructions, company_lexicon=company_lexicon)
    )

    print("5 octobre Chatbot ready. Type your question. Type 'exit' to quit.")
    while True:
        user_query = input("Q: ")
        if user_query.lower().strip() in ["exit", "quit"]:
            break

        nlu_text = parse_query_nlu(user_query)
        augmented_query = f"{user_query}\n\n[NLU Interpretation: {nlu_text}]"
        result = chain.invoke({"question": augmented_query})

        if not result["answer"].strip():
            print("A: I'm not sure. Could you rephrase your question or specify the metric/time period?")
        else:
            print("A:", result["answer"], "\n")


if __name__ == "__main__":
    main()
