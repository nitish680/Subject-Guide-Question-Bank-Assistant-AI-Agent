# import os
# from langchain_core.messages import BaseMessage
# from typing import Annotated,TypedDict
# from langgraph.graph import add_messages
# from Rag import rag_chatbot

# class state:
#     answer:str
#     query:str
#     category:str
#     source_document:list
#     context:str
#     messages:Annotated[list[BaseMessage],add_messages]
# def classify(state:state):
#      prompt = f"""
#     Classify the question.

#     Categories:
#     - academic
#     - general

#     Return only one word.

#     Question:
#     {state["query"]}"""
#      response= self.llm.invoke(prompt)
from langgraph.graph import StateGraph, START, END
from typing import TypedDict


class GraphState(TypedDict):
    query: str
    category: str
    context: str
    answer: str
    source_documents: list


def build_graph(chatbot):

    builder = StateGraph(GraphState)

    # Nodes
    builder.add_node(
        "classifier",
        chatbot.classifier_node)

    builder.add_node(
        "academic_search",
        chatbot.academic_node)

    builder.add_node(
        "web_search",
        chatbot.web_search_node)

    builder.add_node(
        "generate_answer",
        chatbot.answer_node)
    builder.add_edge(
        START,
        "classifier")
    builder.add_conditional_edges(
        "classifier",
        chatbot.router,
        {
            "academic": "academic_search",
            "general": "web_search"
        }
    )
    builder.add_edge(
        "academic_search",
        "generate_answer")

    builder.add_edge(
        "web_search",
        "generate_answer")
    builder.add_edge(
        "generate_answer",
        END)
    return builder.compile()