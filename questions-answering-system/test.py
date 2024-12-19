from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# Define the model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Initialize the pipeline (note the change here)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=0)

# Example usage
question = "What is the capital of France?"
context = "The capital of France is Paris. The city is known for its art, fashion, and history."

# Get the answer
answer = qa_pipeline(question=question, context=context)
print(f"Answer: {answer['answer']}")
print(f"Score: {answer['score']}")
