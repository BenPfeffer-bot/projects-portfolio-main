import os
import sys
import pandas as pd

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA

########################################
# CONFIGURATION
########################################

# Adjust these paths as needed:
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ANALYSIS_DIR = "/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/projects-portfolio-main/5-octobre/data/analysis"

# Path to your local LLM model file (GGML format if using llama-cpp)
# You must download a compatible model, for example a LLaMA 2 7B GGML
model_path = "/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/projects-portfolio-main/5-octobre/models/llama-3.1-8b-instruct.ggmlv3.q4_1.bin"

# Name of the embedding model. We'll use a free one from sentence-transformers
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

########################################
# FUNCTIONS
########################################


def create_analysis_summary_if_needed():
    """
    Check if analysis_summary.md exists. If not, attempt to create it from summary.csv or other CSVs.

    Reasoning:
    - If we haven't already created a textual summary of the analysis, we need to do so now.
    - This summary file is what the chatbot will read and embed.
    - The summary should contain key metrics and insights derived from your analysis pipeline.

    Note:
    - In a real scenario, you'd extract meaningful text from all generated CSV files and combine into a textual narrative.
    - Here, we just check if 'analysis_summary.md' exists and if not, create a basic one from summary.csv if available.
    """
    summary_md = os.path.join(DATA_ANALYSIS_DIR, "analysis_summary.md")
    if os.path.exists(summary_md):
        return summary_md  # Already exists

    summary_csv = os.path.join(DATA_ANALYSIS_DIR, "summary.csv")
    text = "Project Insights Summary:\n\n"
    if os.path.exists(summary_csv):
        df_summary = pd.read_csv(summary_csv)
        for _, row in df_summary.iterrows():
            metric, value = row["metric"], row["value"]
            text += f"- {metric}: {value}\n"
    else:
        text += "No summary.csv found. Please run analysis pipeline first.\n"

    with open(summary_md, "w") as f:
        f.write(text)

    return summary_md


def setup_vectorstore(summary_md):
    """
    Setup the vector store from the analysis summary using free embeddings (no OpenAI).

    Steps:
    1. Load the analysis summary text.
    2. Split into chunks if needed.
    3. Embed using HuggingFace Embeddings.
    4. Store embeddings in Chroma vector database.

    Reasoning:
    - We need vector embeddings to do semantic search on the summary.
    - By using sentence-transformers, we avoid paid API calls.
    - ChromaDB is a lightweight local vector store, easy to set up with LangChain.

    Returns:
    --------
    vectorstore : Chroma
        The vector database initialized with our summary embeddings.
    """
    with open(summary_md, "r") as f:
        docs = f.read()

    # Reasoning on chunking:
    # If docs is large, break it down to chunks to improve retrieval granularity.
    chunk_size = 500
    chunks = [docs[i : i + chunk_size] for i in range(0, len(docs), chunk_size)]

    # Use a free embedding model (HuggingFaceEmbeddings)
    # This does not require API keys and runs locally.
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # Create Chroma vector store
    vectorstore = Chroma.from_texts(chunks, embeddings, collection_name="analysis_collection")
    return vectorstore


def create_qa_chain(vectorstore):
    """
    Create a RetrievalQA chain using a local LLM (via llama-cpp).

    Reasoning:
    - We use LlamaCpp LLM wrapper to load a local model (ggml format).
    - This approach is completely free (aside from compute resources) and private.
    - The RetrievalQA chain will:
      * On each query, retrieve top relevant text chunks.
      * Provide them as context to the local LLM.
      * The LLM will then produce an answer grounded in the retrieved context.

    Returns:
    --------
    qa_chain : RetrievalQA
        The retrieval augmented QA chain.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # LLM configuration:
    # temperature=0 for deterministic answers,
    # max_tokens: adjust as per model constraints,
    # model_path points to your local model.
    llm = LlamaCpp(
        model_path=model_path,
        n_threads=4,  # Adjust threads for performance
        max_tokens=512,  # Limit output length
        temperature=0,
    )

    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa_chain


def query_chatbot(qa_chain, query):
    """
    Query the chatbot with a user query and return the answer.

    Reasoning:
    - Just a wrapper to send a query and print the answer.
    """
    return qa_chain.run(query)


########################################
# MAIN EXECUTION
########################################

if __name__ == "__main__":
    # Reasoning:
    # 1. Ensure we have a summary file. If not, attempt to create it.
    # 2. Setup vector store with free embeddings.
    # 3. Load a free local LLM model with llama-cpp.
    # 4. Interactively query the chatbot.

    summary_md = create_analysis_summary_if_needed()

    if not os.path.exists(summary_md):
        print("No summary found and could not create one. Please run analysis pipeline first.")
        sys.exit(1)

    vectorstore = setup_vectorstore(summary_md)
    qa_chain = create_qa_chain(vectorstore)

    # Example queries
    queries = [
        "What was the total revenue from the last year?",
        "How many unique customers do we have monthly?",
        "Explain the cart abandonment rate and how it's calculated.",
        "Can you forecast the monthly revenue for the next 3 months?",
    ]

    for q in queries:
        print(f"Q: {q}")
        answer = query_chatbot(qa_chain, q)
        print(f"A: {answer}\n")
        # The answer should reflect info from your analysis_summary.md
        # If not sufficient info is found, consider enriching the summary file.
