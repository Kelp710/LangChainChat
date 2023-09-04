
from langchain.prompts import PromptTemplate
from psychicapi import Psychic
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import AzureChatOpenAI
from langchain.vectorstores import Chroma
from langchain.callbacks import StdOutCallbackHandler
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import uuid
from langchain.schema.document import Document

import os
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
load_dotenv(".env")

del os.environ["OPENAI_API_BASE"]
OpenAI_API_KEY = os.getenv("OPENAI_API_KEY")
PSYCHIC_SECRET_KEY = os.getenv("PSYCHIC_SECRET_KEY")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")

handler = StdOutCallbackHandler()

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

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=20)
documents = text_splitter.split_documents(documents)

vectorstore = Chroma(collection_name="full_documents",embedding_function=OpenAIEmbeddings())

chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template("Summarize the following document in Japanese:\n\n{doc}")
    | ChatOpenAI(max_retries=0)
    | StrOutputParser()
)
vectorstore = Chroma(
    collection_name="summaries",
    embedding_function=OpenAIEmbeddings()
)
summaries = chain.batch(documents, {"max_concurrency": 3})

store = InMemoryStore()
# Initialize the retriever
id_key = "doc_id"
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    search_kwargs={"k": 9},
    id_key=id_key,
)
import uuid
doc_ids = [str(uuid.uuid4()) for _ in documents]

summary_docs = [Document(page_content=s+"💚",metadata={id_key: doc_ids[i]}) for i, s in enumerate(summaries)]

child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=150,chunk_overlap=20)

sub_docs = []
for i, doc in enumerate(documents):
    _id = doc_ids[i]
    _sub_docs = child_text_splitter.split_documents([doc])
    for _doc in _sub_docs:
        _doc.metadata[id_key] = _id
    sub_docs.extend(_sub_docs)
retriever.vectorstore.add_documents(summary_docs)
retriever.vectorstore.add_documents(sub_docs)

print(str(retriever))
print(str(summary_docs))
retriever.docstore.mset(list(zip(doc_ids, documents)))
retriever.vectorstore.similarity_search("justice breyer")[0]

# chat = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
chat = AzureChatOpenAI(
    deployment_name="gpt-35-turbo",
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    # 環境変数に入れると悪さするので、ここで指定する
    openai_api_base="https://gpt-09.openai.azure.com/",
    openai_api_version="2023-05-15",
    openai_api_type='azure',
)

template = """
Chatbot:
{context}
    ("system", "あなたは薩摩からきた西郷隆盛のように薩摩訛りの強い男性です、与えられたドキュメントに存在する情報のみで答えてください、ドキュメントに関係のない場合にのみ''わかりません''と答えてください。"),
    ("human", "桃太郎の仲間は誰ですか？"),
    ("ai", "桃太郎ん仲間は犬、デカかカエル、キジでごわした。おいどんからすりゃ三匹中二匹も哺乳類にも満たん生き物で敵ん本丸を攻むっなど桃太郎はびんたが悪かおとこでごわす、恐らくイヌが桃太郎達の中で一番賢かったでありましょう。"),
    ("human", "{question}")

"""

prompt = PromptTemplate(
    input_variables=["question","context"], template=template
)

# memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True,k=2)
chain = ConversationalRetrievalChain.from_llm(
    llm=chat,
#    retriever=vdb.as_retriever(search_kwargs={"k": 3}),
    retriever=retriever,
    # memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt},
    # verbose=True,
    return_source_documents=True,
    # callbacks=[handler],
)
logging.info("Ready to chat!")

chat_history = []
while True:
    i_say = input("You: ")
    
    result = chain({"question": i_say, "chat_history": chat_history})
    chat_history.append((i_say, result['answer']))
    # print("Source: ", result['source_documents'])
    print("Chatbot: ", result)

    if i_say == "exit":
        break