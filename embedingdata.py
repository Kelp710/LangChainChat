
from psychicapi import Psychic
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
import os
from dotenv import load_dotenv
import logging


load_dotenv(".env")
del os.environ["OPENAI_API_BASE"]
PSYCHIC_SECRET_KEY = os.getenv("PSYCHIC_SECRET_KEY")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")
OpenAI_API_KEY = os.getenv("OPENAI_API_KEY")

psychic = Psychic(secret_key=PSYCHIC_SECRET_KEY)
raw_docs = psychic.get_documents(account_id=ACCOUNT_ID).documents
if raw_docs is None:
    raise Exception("No docs found!")
print(
"Generating embeddings from your docs and inserting them into Chroma...")
documents = [
Document(
    page_content=doc["content"],
    metadata={
    "title": doc["title"],
    "source": doc["uri"]
    },
) for doc in raw_docs
]

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=300,
#                                                 chunk_overlap=50)
# texts = text_splitter.split_documents(documents)

# embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",openai_api_key=OpenAI_API_KEY)

# vdb = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=50)
# This text splitter is used to create the child documents
# It should create documents smaller than the parent
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400,chunk_overlap=50)



vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory="./chroma_family_db")
# The storage layer for the parent documents
store = InMemoryStore()

# Initialize the retriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
   
)
retriever.add_documents(documents,None)