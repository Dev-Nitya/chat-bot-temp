import os 
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
from langchain import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.memory import FileChatMessageHistory

openai_api_key = os.environ["OPENAI_API_KEY"]
chatbot = ChatOpenAI(
    model="gpt-4o-mini")

memory = ConversationBufferMemory(
    chat_memory=FileChatMessageHistory("messages.json"),
    return_messages=True,
    memory_key="chat_history",
)

prompt = ChatPromptTemplate(
    input_variables=["content","chat_history"],
    messages=[
        HumanMessagePromptTemplate.from_template("{content}"),
        MessagesPlaceholder(variable_name="chat_history"),
    ]
)

chain = LLMChain(
    llm=chatbot,
    prompt=prompt,
    memory=memory
)

response = chain.invoke({"content": "Hello"})
response = chain.invoke({"content": "My name is Nitya"})
response = chain.invoke({"content": "What is my name?"})

print(response)
# messages_to_the_chatbot = [
#     HumanMessage(content="My favorite color is black")
# ]

# response = chatbot.invoke(messages_to_the_chatbot)

# print(response.content)