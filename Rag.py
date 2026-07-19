import os 
from dotenv import load_dotenv
import streamlit as st

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
        self.groq_api_key = st.secrets["groq_api_key"]

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
    
    def data_pipline(self,path):
        try:
            print("fetching document")

            loader=PyPDFLoader(path)
            doc=loader.load()
            print("document load sucessfuly")
       
        except Exception as e:
            print(f"Could not read PDF: {e}")
    
    
        
        if not doc:
            raise ValueError("document is not available")
        
        splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks=splitter.split_documents(doc)

        print("sucessfully..........")
        if not chunks:
            raise ValueError("No readable text found in the uploaded PDF.")
        # embedding=HuggingFaceEmbeddings()
        self.vector_store=Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding,

        )
        self.retriever=self.vector_store.as_retriever(search_kwargs={"k":2})
        # print("vector store created secussfuly")
    
        # llm=ChatGroq(
        #     model="openai/gpt-oss-20b",
        #     api_key=self.groq_api_key,
        #     temperature=0.5,
        #     model_kwargs={'tool_choice':'auto'}
        # )
        self.qa_chain=RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True

        )
    def ask_query(self,query):
        response=self.qa_chain.invoke({'query':query})
        return {'result':response['result'],
                'source':response['source_documents']}
    
    
    

# if __name__=='__main__':
#     # rag=rag_chatbot()
#     # docs=rag.load_document('C:\RAG_RANDOM\scholarship_info (1).pdf')
#     # chunks=rag.splitter(docs)
#     rag.vector_store(chunks)
#     rag.llm()
#     rag.ask_query("who is eligible for scholarship ")
    