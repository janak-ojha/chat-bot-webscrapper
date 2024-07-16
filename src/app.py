import os
import traceback
import streamlit as st
from utils.get_urls import Scrape_urls
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import chroma
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import huggingface_hub
import PyPDF2
from langchain_core.documents import Document

load_dotenv()
HF_TOKEN = os.getenv('HUGGINGFACE_API_KEY')

def get_vectorstore_from_url(url, max_depth, pdf_text=None):
    try:
        if not os.path.exists('src/chroma'):
            os.makedirs('src/chroma')
        if not os.path.exists('src/scrape'):
            os.makedirs('src/scrape')

        documents = []
        if url:
            urls = Scrape_urls(url, max_depth)
            loader = WebBaseLoader(urls)
            documents.extend(loader.load())

        if pdf_text:
            documents.append(Document(page_content=pdf_text, metadata={"source": "uploaded_pdf"}))

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=56, chunk_overlap=0)
        document_chunks = text_splitter.split_documents(documents)

        # Creating embedding
        embedding = HuggingFaceInferenceAPIEmbeddings(
            api_key=HF_TOKEN,
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        vector_store = chroma.from_documents(document_chunks, embedding)
        return vector_store, len(documents)
    except Exception as e:
        st.error(f"Error occurred during processing: {e}")
        traceback.print_exc()
        return None, 0


#define function to create context retriver chain
def get_context_retriever_chain(vector_store):
    llm = huggingface_hub(
        repo_id="HuggingFaceH4/zephyr-7b-alpha",
        model_kwargs = {"temperature":0.8,"max_new_tokens":512,"max_length":64},
    )
    retriever = vector_store.as_retriever();
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ( "user","{input}"),
        ("user","Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain =create_history_aware_retriever(llm,retriever,prompt)
    return retriever_chain

#define function to create  conversational RAG chain
def get_conversational_rag_chain(retriever_chain):
    