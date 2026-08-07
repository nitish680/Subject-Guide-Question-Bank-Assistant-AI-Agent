import sys
import asyncio
import os

import streamlit as st
import uuid
import tempfile
from langchain_core.messages import HumanMessage, AIMessage
from backend import chatbot, extract_id

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Session State Helpers
def generate_thread_id():
    return str(uuid.uuid4())

def reset():
    new_id = generate_thread_id()
    st.session_state['thread_id'] = new_id
    if st.session_state['thread_id'] not in st.session_state['thread_id_history']:
        st.session_state['thread_id_history'].append(new_id)
    st.session_state['message_history'] = []
    st.session_state['processed_files'] = set()  # Reset uploaded files set

def load_conversation(thread_id):
    config = {"configurable": {"thread_id": str(thread_id)}}
    state = chatbot.get_state(config=config)
    if not state.values:
        return []
    return state.values.get("messages", [])

def data_ingestion(file_bytes, filename, thread_id):
    """Processes PDF and stores vectors under a thread-specific Chroma collection with file source metadata."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(file_bytes)
        path = temp_file.name

    try:
        loader = PyPDFLoader(path)
        docs = loader.load()
        
        # Attach original filename to document metadata
        for doc in docs:
            doc.metadata["source_file"] = filename

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Ingest chunks into thread-specific collection
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=str(thread_id),
            persist_directory="./chroma_db"
        )
    finally:
        if os.path.exists(path):
            os.remove(path)

# Session state initialization
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'thread_id_history' not in st.session_state:
    st.session_state['thread_id_history'] = extract_id()

if 'processed_files' not in st.session_state:
    st.session_state['processed_files'] = set()

if st.session_state['thread_id'] not in st.session_state['thread_id_history']:
    st.session_state['thread_id_history'].append(st.session_state['thread_id'])

# UI Sidebar Configuration
st.sidebar.title("AI ChatBot")

# Accept multiple PDF uploads
uploaded_files = st.sidebar.file_uploader(label='Upload PDFs', type='pdf', accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_id = f"{st.session_state['thread_id']}_{uploaded_file.name}_{uploaded_file.size}"
        
        if file_id not in st.session_state['processed_files']:
            with st.sidebar.spinner(f"Ingesting {uploaded_file.name}..."):
                data_ingestion(uploaded_file.read(), uploaded_file.name, st.session_state['thread_id'])
                st.session_state['processed_files'].add(file_id)
                st.sidebar.success(f"Added: {uploaded_file.name}")

if st.sidebar.button('New Chat'):
    reset()

st.sidebar.title("Chat History")
for t_id in st.session_state['thread_id_history']:
    if st.sidebar.button(str(t_id), key=f"btn_{t_id}"):
        st.session_state['thread_id'] = t_id
        st.session_state['processed_files'] = set()  # Reset tracker on thread switch
        messages = load_conversation(t_id)
        temp = []
        for message in messages:
            if message.content and isinstance(message.content, str):
                role = 'user' if isinstance(message, HumanMessage) else 'ai'
                temp.append({'role': role, 'content': message.content})
        st.session_state['message_history'] = temp

# Runtime Configuration for LangGraph state persistence
confi = {"configurable": {"thread_id": str(st.session_state['thread_id'])}}

# Render active chat history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# User input & Streaming execution
user = st.chat_input("Ask anything...")

if user:
    st.session_state['message_history'].append({'role': 'user', 'content': user})
    with st.chat_message('user'):
        st.markdown(user)

    with st.chat_message('ai'):
        def AIstreamonly():
            for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user)]},
                config=confi,
                stream_mode='messages'
            ):
                if (
                    metadata.get("langgraph_node") == "chat_with"
                    and isinstance(message_chunk, AIMessage)
                    and message_chunk.content
                ):
                    yield message_chunk.content

        ai_messages = st.write_stream(AIstreamonly())

    st.session_state['message_history'].append({'role': 'ai', 'content': ai_messages})