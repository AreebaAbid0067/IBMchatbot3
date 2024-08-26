from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')


# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("F:\ML_chatbot\MLchatbot\MLchatbot\data")

text_chunks = text_split(extracted_data)

embeddings = download_hugging_face_embeddings()

from pinecone import Pinecone

pc = Pinecone(api_key="c3e6725e-a8b9-4c60-84cb-d7a08d467126")
index = pc.Index("mlchatbot")

#Creating Embeddings for Each of The Text Chunks & storing
from langchain_pinecone import PineconeVectorStore

vector_store = PineconeVectorStore(index=index, embedding=embeddings)




