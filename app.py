import tempfile
import streamlit as st
from Rag import rag_chatbot

st.title("📄AI-Chatbot")
st.markdown("An AI-powered Academic Learning Assistant that helps students learn from lecture notes, textbooks, lab manuals, and previous-year question papers using Retrieval-Augmented Generation (RAG)")
# Create chatbot only once
if "rag_chat" not in st.session_state:
    st.session_state.rag_chat = rag_chatbot()

rag_chat = st.session_state.rag_chat

# Track whether documents have already been processed
if "processed_files" not in st.session_state:
    st.session_state.processed = []
#new changes
if "message_history" not in st.session_state:
    st.session_state['message_history']=[]
    
uploaded_files = st.file_uploader(
    "Upload PDF(s)",
    type=["pdf"],
    accept_multiple_files=True
)


# Process documents only once after upload
if uploaded_files and not st.session_state.processed:

    pdf_files = []

    for uploaded_file in uploaded_files:
        source=uploaded_file.name
        documenttype=st.selectbox(f"select document type for {uploaded_file.name}",['Notes','Textbook','PYQ','lab-manual'],key=uploaded_file.name)
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(uploaded_file.read())
            pdf_files.append({'path':tmp.name,
                              'source':source,
                              'document_type':documenttype})
            # pdf_paths.append(tmp.name)

    rag_chat.data_pipline(pdf_files)

    st.session_state.processed = True

    st.success("PDFs uploaded successfully!")

# for message in st.session_state['message_history']:
#      with st.chat_message(message['role']):
#          st.text(message['result'])
#          for doc in message["source"]:
#                      # st.write(doc.metadata)
#                          source = doc.metadata["source"]
#                          page = doc.metadata["page"] + 1
#                          st.write(f"📄 {source} (Page {page})")
for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):

        st.text(message["result"])

        if message["role"] == "ai":

            displayed_sources = set()

            st.subheader("📚 Sources")

            for doc in message["source"]:

                source = doc.metadata["source"]
                page = doc.metadata["page"] + 1

                key = (source, page)

                if key not in displayed_sources:

                    displayed_sources.add(key)

                    st.write(f"📄 {source} (Page {page})")

question = st.chat_input("Ask your question")
# question=st.text_input("ASK Anything..")

# if st.button("Ask"):
# if question:
    
#     if not st.session_state.processed:
#         st.warning("Please upload PDF(s) first.")

#     elif question:


#             # answer = rag_chat.ask_query(question)
#         st.session_state['message_history'].append({'role':'user','result':question})
#         with st.chat_message('user'):
#             st.text(question)

#         # st.write(answer["result"])
#         with st.spinner("Thinking..."):
#             answer=rag_chat.ask_query(question)
#             with st.chat_message('ai'):
#                 st.text(answer['result'])
#             st.session_state['message_history'].append({'role':'ai','result':answer,'source':answer['source']})

#             st.subheader("Sources")
        
#             displayed_sources = set()

#             for doc in answer["source"]:
#             # st.write(doc.metadata)
#                 source = doc.metadata["source"]
#                 page = doc.metadata["page"] + 1

#                 key = (source, page)
#                 if key not in displayed_sources:

#                     displayed_sources.add(key)

#                     st.write(f"📄 {source} (Page {page})")
if question:

    if not st.session_state.processed:
        st.warning("Please upload PDFs first.")

    else:

        # User message
        st.session_state["message_history"].append(
            {
                "role": "user",
                "result": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.spinner("Thinking..."):

            answer = rag_chat.ask_query(question)

        with st.chat_message("ai"):

            st.markdown(answer["result"])

            st.markdown("### 📚 Sources")

            displayed_sources = set()

            for doc in answer["source"]:

                source = doc.metadata["source"]
                page = doc.metadata["page"] + 1

                key = (source, page)

                if key not in displayed_sources:

                    displayed_sources.add(key)

                    st.write(f"📄 {source} (Page {page})")

        st.session_state["message_history"].append(
            {
                "role": "ai",
                "result": answer["result"],
                "source": answer["source"]
            }
        )