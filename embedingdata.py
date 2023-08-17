
from psychicapi import Psychic
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os
from dotenv import load_dotenv


def get_db():
    load_dotenv('.env')
    PSYCHIC_SECRET_KEY = os.getenv("PSYCHIC_SECRET_KEY")
    ACCOUNT_ID = os.getenv("ACCOUNT_ID")

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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                    chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=os.getenv("OPENAI_API_KEY"))

    vdb = Chroma.from_documents(texts, embeddings)

    return vdb

print(get_db())