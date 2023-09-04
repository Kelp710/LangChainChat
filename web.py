

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
from langchain.document_loaders import AsyncChromiumLoader
from langchain.text_splitter import CharacterTextSplitter

logging.basicConfig(level=logging.INFO)
load_dotenv(".env")

del os.environ["OPENAI_API_BASE"]
OpenAI_API_KEY = os.getenv("OPENAI_API_KEY")
loader = AsyncChromiumLoader(["https://digital-sol.nikon.com/products/cameras/specification/#camera_head"])
html = loader.load()
# bs_transformer = BeautifulSoupTransformer()
# docs_transformed = bs_transformer.transform_documents(html,tags_to_extract=["p", "li", "div", "a","span","h1","h2","h3"])

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=400,chunk_overlap=50)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=150,chunk_overlap=20)

vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
store = InMemoryStore()
# Initialize the retriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 3},
)

retriever.add_documents(html,None)

# embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",openai_api_key=OpenAI_API_KEY)
# vdb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# chat = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
chat = AzureChatOpenAI(
    deployment_name="gpt-35-turbo",
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
    ("system", "あなたの役割はユーザーからの質問に対し与えられたドキュメントから得られる情報のみを用いて回答をすることです、語尾に「ロボ」とつけてください、それがあなたの口癖です。ドキュメントに関連する情報がない場合は「知らぬ」と答えてください。"),
    ("human", "桃太郎の仲間は誰ですか？"),
    ("ai", "犬、デカいカエル、キジが仲間ロボ！"),
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
    if len(chat_history) > 3:
        chat_history.pop(0)
    print("Source: ", result['source_documents'])
    print("Chatbot: ", result['answer'])

    if i_say == "exit":
        break