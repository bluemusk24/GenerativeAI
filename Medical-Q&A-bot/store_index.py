from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

#print(PINECONE_API_KEY)

extracted_data = load_pdf("data")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Initializing the Pinecone Vectordb

index_name="medical-chatbot"

#Creating Embeddings for Each of The Text Chunks & storing in a Pineconedb
docsearch = PineconeVectorStore.from_texts([t.page_content for t in text_chunks], embedding=embeddings, index_name=index_name)