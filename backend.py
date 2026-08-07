import os
import sqlite3
import requests
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, START, add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()
import streamlit as st

groq_key = os.getenv("groq_api_key") or st.secrets.get("groq_api_key")
weather_key = os.getenv("weather_api_key") or st.secrets.get("weather_api_key")
# Setup SQLite Checkpointer
conn = sqlite3.connect('sqldb', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# LLM Initialization
model_ = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=groq_key,
    temperature=0.5
)

# Tool 1: Weather Tool
@tool
def wheathertool(city: str):
    """Fetches current weather data for a given city."""
    weather_api = weather_key
    url = f'https://api.weatherstack.com/current?access_key={weather_api}&query={city}'
    response = requests.get(url)
    return response.json()

# Tool 2: Web Search Tool
search_tool = DuckDuckGoSearchRun()

# Tool 3: RAG Tool (Updated for Multi-Document + Sources)
@tool
def ragtool(query: str, config: RunnableConfig) -> str:
    """Retrieves context from uploaded PDF documents relevant to the query and includes document sources."""
    try:
        thread_id = config.get("configurable", {}).get("thread_id", "default_thread")
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        vector_store = Chroma(
            collection_name=str(thread_id),
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
        
        docs = vector_store.similarity_search(query, k=4)
        if not docs:
            return "No relevant information found in uploaded documents."
            
        context_parts = []
        for i, doc in enumerate(docs):
            source_file = doc.metadata.get("source_file", "Unknown Document")
            page_num = doc.metadata.get("page", 0) + 1  # 0-indexed to 1-indexed
            
            excerpt = (
                f"Excerpt {i+1} [Source: {source_file}, Page {page_num}]:\n"
                f"{doc.page_content}"
            )
            context_parts.append(excerpt)
            
        return "\n\n".join(context_parts)
    except Exception as e:
        return f"Error retrieving context: {str(e)}"

# Register all tools
tools = [search_tool, wheathertool, ragtool]
llm_with_tools = model_.bind_tools(tools=tools)

# Define Graph State
class chat_state(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

tool_node = ToolNode(tools=tools)

def chat_with_llm(state: chat_state):
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {'messages': [response]}

# Build LangGraph Loop
graph = StateGraph(chat_state)
graph.add_node("chat_with", chat_with_llm)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_with")
graph.add_conditional_edges("chat_with", tools_condition)
graph.add_edge("tools", "chat_with")

chatbot = graph.compile(checkpointer=checkpointer)

def extract_id():
    thread_list = set()
    for checkpoint in checkpointer.list(None):
        thread_list.add(checkpoint.config['configurable']['thread_id'])
    return list(thread_list)