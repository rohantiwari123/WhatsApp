import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
try:
    chat_model = ChatGroq(temperature=0.7, groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
    response = chat_model.invoke([HumanMessage(content="Hi")])
    print(f"Success: {response.content}")
except Exception as e:
    print(f"Error: {e}")
