from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
import sys
import time
import requests

DB_FAISS_PATH = "vectorstore/db_faiss"
file_path = "data/companyreview_dataset.csv"

# Load data
start_time = time.time()
loader = CSVLoader(file_path, encoding="utf-8", csv_args={'delimiter': ','})
data = loader.load()
print(f"Data loaded in {time.time() - start_time} seconds.")

# Split the text into chunks
start_time = time.time()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
text_chunks = text_splitter.split_documents(data)
print(f"Text split into {len(text_chunks)} chunks in {time.time() - start_time} seconds.")

# Download Sentence Transformers Embedding From Hugging Face
start_time = time.time()
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
print(f"Embeddings model loaded in {time.time() - start_time} seconds.")

# Converting the text chunks into embeddings and saving the embeddings into FAISS Knowledge Base
start_time = time.time()
docsearch = FAISS.from_documents(text_chunks, embeddings)
docsearch.save_local(DB_FAISS_PATH)
print(f"Embeddings calculated and FAISS index saved in {time.time() - start_time} seconds.")

vector_store = docsearch
retriever = vector_store.as_retriever()
llm = Ollama(model="llama3:latest")

qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)

# Streamlit interface
st.title("InsightBot: Comprehensive Company Insights Chatbot")
st.write("""
Discover everything you need to know about any company, from ratings and reviews to location and employee count. Type your question and press Enter to get comprehensive insights instantly.
""")

chat_history = []

query = st.text_input("Input Prompt", "")

if st.button("Submit"):
    if query:
        try:
            start_time = time.time()
            result = qa({"question": query, "chat_history": chat_history})
            response = result['answer']
            st.write("Response: ", response)
            st.write(f"Response time: {time.time() - start_time} seconds.")
        except requests.ConnectionError as e:
            st.error(f"Connection error: {e}")
        except requests.HTTPError as e:
            st.error(f"HTTP error: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
