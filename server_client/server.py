from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import csv
import os

app = FastAPI()

DB_FAISS_PATH = "vectorstore/db_faiss"
file_path = "data/companyreview_dataset.csv"

# Verify the number of rows in the CSV file
with open(file_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    row_count = sum(1 for row in reader) - 1  # subtract 1 for header row
    print(f"Total number of rows in the CSV file: {row_count}")

# Load data
start_time = time.time()
loader = CSVLoader(file_path, encoding="utf-8", csv_args={'delimiter': ','})
data = loader.load()
print(f"Data loaded in {time.time() - start_time} seconds.")
print(f"Number of rows loaded: {len(data)}")

# Split the text into chunks
start_time = time.time()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
text_chunks = text_splitter.split_documents(data)
print(f"Text split into {len(text_chunks)} chunks in {time.time() - start_time} seconds.")

# Print out a few chunks to inspect
for i, chunk in enumerate(text_chunks[:5]):
    print(f"Chunk {i}: {chunk.page_content[:500]}")  # Print the first 500 characters of each chunk

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

class QueryRequest(BaseModel):
    question: str
    chat_history: list = []

class QueryResponse(BaseModel):
    answer: str
    response_time: float

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    start_time = time.time()
    
    # Custom logic for specific queries
    question_lower = request.question.lower()
    if "total number of companies" in question_lower or "how many companies" in question_lower:
        # Directly count the companies
        total_companies = len(data)
        answer = f"The total number of companies listed in the dataset is {total_companies}."
        print(f"Custom logic invoked for question: {request.question}")
    else:
        # Use the conversational retriever for other queries
        result = qa({"question": request.question, "chat_history": request.chat_history})
        answer = result['answer']
    
    response_time = time.time() - start_time
    return QueryResponse(answer=answer, response_time=response_time)

if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        print(f"Error: {e}")
        os.system("lsof -ti:8000 | xargs kill -9")  # Free up port 8000 if occupied
        uvicorn.run(app, host="0.0.0.0", port=8000)
