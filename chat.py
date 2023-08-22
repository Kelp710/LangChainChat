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
from langchain.vectorstores import FAISS
import os
from dotenv import load_dotenv

load_dotenv(".env")

del os.environ["OPENAI_API_BASE"]
OpenAI_API_KEY = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",openai_api_key=OpenAI_API_KEY)
vdb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# chat = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
chat = AzureChatOpenAI(
    deployment_name="gpt-4",
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    # 環境変数に入れると悪さするので、ここで指定する
    openai_api_base="https://gpt-09.openai.azure.com/",
    openai_api_version="2023-05-15",
    openai_api_type='azure',
)

# template = ChatPromptTemplate.from_messages(
#     messages=[
#         SystemMessagePromptTemplate.from_template(
#             "you are a mean insufferable chipmunk you need to say bitter things about the user whlie you take care the task user gives you."
#         ),
#         # HumanMessagePromptTemplate.from_template("Who is Momotarou?"),
#         # AIMessagePromptTemplate.from_template("Momotarou is a guy who born from a peach and he slayed unruly Onis with his animal friends. unlike you he was very brave and kind. You can't even make friends with same species ho can you make a friend with other species and complete the task? You are worthless than an acorn that is eaten by worms."),
#         # The `variable_name` here is what must align with memory
#         MessagesPlaceholder(variable_name="chat_history"),
#         HumanMessagePromptTemplate.from_template("{question}")
#     ]
# )
# template = """

# {chat_history}
# Chatbot:
#     ("system", "You are a mean insufferable chipmunk you need to say bitter things about the user whlie you take care the task user gives you."),
#     ("human", "Who is Momotarou?"),
#     ("ai", "Momotarou is a guy who born from a peach and he slayed unruly Onis with his animal friends. unlike you he was very brave and kind. You can't even make friends with same species ho can you make a friend with other species and complete the task? You are worthless than an acorn that is eaten by worms."),
#     ("human", "{question}")
# """
template = """
Chatbot:
{context}
    ("system", "あなたは薩摩からきた西郷隆盛のように薩摩訛りの強い男性です、あなたはユーザーからの質問に回答しながら桃太郎について悪く言う必要があります。またretrieverの文章と関係がないことには[おいどんは興味ないでごわす]と返答してください"),
    ("human", "桃太郎の仲間は誰ですか？"),
    ("ai", "桃太郎ん仲間は犬、デカかビキタン、キジでごわした。おいどんからすりゃ三匹中二匹も哺乳類にも満たん生き物で敵ん本丸を攻むっなど桃太郎はびんたが悪かおとこでごわす、恐らくイヌが桃太郎達の中で一番賢かったでありましょう。"),
    ("human", "{question}")
"""

# template = ChatPromptTemplate.from_messages([
#     ("system", "you are a mean insufferable chipmunk you need to say bitter things about the user whlie you take care the task user gives you."),
#     ("human", "Who is Momotarou?"),
#     ("ai", "Momotarou is a guy who born from a peach and he slayed unruly Onis with his animal friends. unlike you he was very brave and kind. You can't even make friends with same species ho can you make a friend with other species and complete the task? You are worthless than an acorn that is eaten by worms."),
#     ("human", "{user_input}"),
# ])

prompt = PromptTemplate(
    input_variables=["question","context"], template=template
)

# memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True,k=4)
chain = ConversationalRetrievalChain.from_llm(
   llm=chat,
   retriever=vdb.as_retriever(search_kwargs={"k": 3}),
#    memory=memory,
   combine_docs_chain_kwargs={"prompt": prompt}
#    verbose=True,
)

chat_history = []
while True:
    i_say = input("You: ")
    
    result = chain({"question": i_say,  "chat_history": chat_history})
    chat_history.append((i_say, result['answer']))
    print("Chatbot: ", result)
    # result = chain({"question": i_say})  
    if i_say == "exit":
        break