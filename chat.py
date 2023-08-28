from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)
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

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=50)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=50)

vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
store = InMemoryStore()
# Initialize the retriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 7},
)
retriever.add_documents(documents,None)

# embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",openai_api_key=OpenAI_API_KEY)
# vdb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# chat = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
chat = AzureChatOpenAI(
    deployment_name="gpt-4",
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    # 環境変数に入れると悪さするので、ここで指定する
    openai_api_base="https://gpt-09.openai.azure.com/",
    openai_api_version="2023-05-15",
    openai_api_type='azure',
)
# template = "You are a helpful assistant that translates {input_language} to {output_language}."
# systemMessagePrompt = SystemMessagePromptTemplate.from_template(template)
# humanTemplate = "{question}"
# humanMessagePrompt = HumanMessagePromptTemplate.from_template(humanTemplate)

# prompt = PromptTemplate(
#      input_variables=["question","context",'input_language'], template=template
# )
# systemMessagePrompt2 = SystemMessagePromptTemplate({
#   prompt,
# })

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
    print("Source: ", result['source_documents'])
    print("Chatbot: ", result['answer'])

    if i_say == "exit":
        break