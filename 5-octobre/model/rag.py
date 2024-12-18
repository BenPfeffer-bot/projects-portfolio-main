import os
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch.nn.functional as F

# Paths
BASE_DIR = os.path.dirname(__file__)
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vector_store.faiss")
CHUNKS_PATH = os.path.join(BASE_DIR, "chunks.txt")

# Load model from HuggingFace for embeddings
tokenizer_embedding = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)
model_embedding = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index(VECTOR_STORE_PATH)

# Load chunks
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    raw = f.read()
chunks = [c.strip() for c in raw.split("------") if c.strip()]

# Load GPT-2 model and tokenizer (much smaller, easily downloadable model)
MODEL_NAME = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

if torch.cuda.is_available():
    model = model.cuda()
else:
    model = model.to("cpu")


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def get_embedding(text):
    """Get embeddings for a given text using the embedding model."""
    encoded_input = tokenizer_embedding(
        text, padding=True, truncation=True, return_tensors="pt", max_length=512
    )
    with torch.no_grad():
        model_output = model_embedding(**encoded_input)
    embedding = mean_pooling(model_output, encoded_input["attention_mask"])
    return embedding[0].numpy()


def retrieve_chunks(query, k=3):
    """
    Given a user query, embed it and retrieve top-k relevant chunks from the vector store.
    """
    query_embedding = get_embedding(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    retrieved = [chunks[i] for i in indices[0]]
    return retrieved


def build_prompt(user_query, retrieved_chunks):
    """
    Build a prompt for the LLM.
    The prompt includes instructions and the retrieved context, followed by the user query.
    """
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""You are a helpful assistant with knowledge about the company's metrics. Use the following context to answer the user's question as accurately as possible.
        Context: {context}
        User: {user_query} Assistant:"""

    # prompt = f"""You are a helpful assistant with knowledge about the company's metrics. Use the following context to answer the user's question as accurately as possible.
    # Context: {context}
    # User: {user_query} Assistant:"""
    return prompt.strip()


def generate_answer(prompt, max_tokens=256):
    """
    Generate an answer from the Mistral model given a prompt.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate with specified parameters
    output = model.generate(
        **inputs,
        max_length=inputs.input_ids.shape[1] + max_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
    )

    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract just the assistant's response
    if "Assistant:" in answer:
        answer = answer.split("Assistant:")[-1].strip()

    return answer


def answer_query(user_query, k=3):
    """
    Main function to answer a user query using RAG.
    """
    retrieved = retrieve_chunks(user_query, k=k)
    prompt = build_prompt(user_query, retrieved)
    answer = generate_answer(prompt)
    return answer


if __name__ == "__main__":
    # Test with a sample query
    test_query = "What is the cart abandonment rate?"
    ans = answer_query(test_query)
    print("Q:", test_query)
    print("A:", ans)
