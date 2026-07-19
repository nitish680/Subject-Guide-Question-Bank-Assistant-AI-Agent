import streamlit as st
from Rag import rag_chatbot

st.title("📄 AI ChatBot")

# Create chatbot only once
if "rag_chat" not in st.session_state:
    st.session_state.rag_chat = rag_chatbot()

rag_chat = st.session_state.rag_chat

# uploaded_file = st.file_uploader(
#     "Upload your PDF",
#     type=["pdf"]
# )

# Process PDF only once
import tempfile

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    try:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:

            tmp.write(uploaded_file.read())

            pdf_path = tmp.name

        rag_chat.data_pipline(pdf_path)

        st.session_state.processed = True
        st.success("PDF uploaded successfully!")
    except Exception as e:
        st.write("ERROR {e}....")

question = st.text_input("Ask your question")

if st.button("Ask"):

    if question:

        with st.spinner("Thinking..."):
            answer = rag_chat.ask_query(question)

        st.write(answer['result'])
        # st.write(answer['source'])