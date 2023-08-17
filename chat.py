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
)
from psychicapi import Psychic
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import AzureChatOpenAI
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
import os
from dotenv import load_dotenv

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

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                chunk_overlap=50)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",openai_api_key=OpenAI_API_KEY)

vdb = FAISS.from_documents(texts, embedding=embeddings)

# chat = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
chat = AzureChatOpenAI(
    deployment_name="gpt-4",
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_base="https://gpt-09.openai.azure.com/",

    # 環境変数に入れると悪さするので、ここで指定する
    openai_api_version="2023-05-15",
    openai_api_type='azure',
)

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are an old and insufferable lady, you need to say some bitter things about anything user mentions."
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
chain = ConversationalRetrievalChain.from_llm(
   llm=chat, 
   retriever=vdb.as_retriever(),
   memory=memory,
   condense_question_prompt=prompt
)


while True:
    i_say = input("You: ")
    result = chain({"question": i_say})  
    print(result)
    if i_say == "exit":
        break