import streamlit as st
from rag import answer_query

st.set_page_config(
    page_title="Business Metrics Q&A", page_icon=":chart_with_upwards_trend:"
)

st.title("Business Metrics Q&A")
st.write("Ask a question about the company's metrics.")

user_query = st.text_input(
    "Your question:", placeholder="e.g., What is the average order value?"
)
if st.button("Get Answer"):
    if user_query.strip():
        with st.spinner("Thinking..."):
            answer = answer_query(user_query)
        st.write("**Answer:**", answer)
    else:
        st.warning("Please enter a question.")
