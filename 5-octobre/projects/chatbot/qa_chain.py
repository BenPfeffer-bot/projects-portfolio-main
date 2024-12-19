# chatbot/qa_chain.py
import os
from langchain.llms import Mistral
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

OUTPUT_DIR = "chatbot/vectorstore"


def main():
    llm = Mistral(model_name="mistral-large-16k-instruct", temperature=0)
    vectorstore = Chroma(collection_name="analysis_collection", persist_directory=OUTPUT_DIR)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    print("Chatbot ready. Type your questions. Type 'exit' to quit.")
    while True:
        query = input("Q: ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = qa_chain.run(query)
        print("A:", answer)


if __name__ == "__main__":
    main()
