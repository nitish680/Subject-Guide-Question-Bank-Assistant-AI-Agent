import os 
from dotenv import load_dotenv
import streamlit as st
from graph import build_graph
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
# from sentence_transformers import SentenceTransformer


class rag_chatbot:
    
    def __init__(self):
        print("loading api")
        load_dotenv()
        self.groq_api_key=os.getenv('groq_api_key')
        # self.groq_api_key = st.secrets["groq_api_key"]

        if not self.groq_api_key :
            raise ValueError("key is not found")
        print("api key fetch secussfuly")
        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
        self.llm=ChatGroq(
            model="openai/gpt-oss-20b",
            api_key=self.groq_api_key,
            temperature=0.5,
            model_kwargs={'tool_choice':'auto'}
        )
        self.search_tool = DuckDuckGoSearchRun()
        self.graph = build_graph(self)
    def data_pipline(self,pdf_files):
        
        all_documents=[]
        for file_info in pdf_files:
            try:
                print("fetching document")

                loader=PyPDFLoader(file_info["path"])
                doc=loader.load()
                for document in doc:
                    document.metadata['source']=file_info['source']
                    document.metadata["document_type"] = file_info["document_type"]
                all_documents.extend(doc)
                print("document load sucessfuly")
       
            except Exception as e:
                print(f"Could not read PDF: {e}")
    
    
        
        if not all_documents:
            raise ValueError("document is not available")
        
        splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks=splitter.split_documents(all_documents)

        print("sucessfully..........")
        if not chunks:
            raise ValueError("No readable text found in the uploaded PDF.")
        # embedding=HuggingFaceEmbeddings()
        self.vector_store=Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding,

        )
        self.retriever=self.vector_store.as_retriever(search_kwargs={"k":2})
        
    def ask_query(self, query):
        state = {
            "query": query,
            "category": "",
            "context": "",
            "answer": "",
            "source_documents": []
        }
        result = self.graph.invoke(state)
        return {
            "result": result["answer"],
            "source": result["source_documents"]
        }
    def router(self, state):
        if state["category"] == "academic":
            return "academic"

        return "general"
    def classifier_node(self, state):

        prompt = f"""
        Classify the following question.

        Categories:
        - academic
        - general

        Return only one word.

        Question:
        {state["query"]}
        """
        response = self.llm.invoke(prompt)
        state["category"] = response.content.strip().lower()
        return state
    def academic_node(self, state):
        docs = self.retriever.invoke(state["query"])
        context = ""
        for doc in docs:
            context += doc.page_content + "\n\n"
        state["context"] = context
        state["source_documents"] = docs

        return state
    def web_search_node(self, state):

        result = self.search_tool.invoke(
            state["query"]
        )

        state["context"] = result
        state["source_documents"] = []

        return state
    def answer_node(self, state):

        prompt = f"""
        You are an AI Academic Learning Assistant.
        Answer ONLY using the provided context.
        If the answer is not available in the context,
        say:
        "I couldn't find this information in the uploaded documents."
    Explain clearly.
    Use headings when needed.
    Context:
    {state["context"]}
    Question:
    {state["query"]}
    """
        response = self.llm.invoke(prompt)
        state["answer"] = response.content
        return state
    