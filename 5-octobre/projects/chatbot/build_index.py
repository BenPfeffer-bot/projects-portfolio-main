# chatbot/build_index.py
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Path to analysis summary
ANALYSIS_SUMMARY_PATH = "/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/projects-portfolio-main/5-octobre/data/analysis/analysis_summary.md"
OUTPUT_DIR = "/Users/benpfeffer/Library/Mobile Documents/com~apple~CloudDocs/projects-portfolio-main/5-octobre/projects/chatbot/vectorstore"


def main():
    with open(ANALYSIS_SUMMARY_PATH, "r") as f:
        doc = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(doc)

    embedding = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    vectorstore = Chroma.from_texts(chunks, embedding, collection_name="analysis_collection", persist_directory=OUTPUT_DIR)
    vectorstore.persist()


if __name__ == "__main__":
    main()
