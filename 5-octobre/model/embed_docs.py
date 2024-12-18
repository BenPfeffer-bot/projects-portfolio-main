import os
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

# Paths
DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
VECTOR_STORE_PATH = os.path.join(os.path.dirname(__file__), "vector_store.faiss")
METRICS_FILE = os.path.join(DOCS_DIR, "metrics_knowledge_base.md")


def load_document(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text, chunk_size=512):
    lines = text.split("\n")
    chunks = []
    current_chunk = []

    for line in lines:
        if len(" ".join(current_chunk + [line])) > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [line]
        else:
            current_chunk.append(line)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


if __name__ == "__main__":
    # Load document
    text = load_document(METRICS_FILE)
    chunks = chunk_text(text, chunk_size=512)

    # Load model from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # Compute embeddings
    embeddings = []
    for chunk in chunks:
        encoded_input = tokenizer(
            chunk, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        with torch.no_grad():
            model_output = model(**encoded_input)
        embedding = mean_pooling(model_output, encoded_input["attention_mask"])
        embeddings.append(embedding[0].numpy())

    embeddings = np.array(embeddings)

    # Create and save FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, VECTOR_STORE_PATH)

    # Save chunks separately
    with open(
        os.path.join(os.path.dirname(__file__), "chunks.txt"), "w", encoding="utf-8"
    ) as f:
        for chunk in chunks:
            f.write(chunk + "\n------\n")
