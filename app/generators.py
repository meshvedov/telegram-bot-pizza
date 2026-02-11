import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from typing import List 
from pydantic import BaseModel, Field
# import whisper # Локальный STT

load_dotenv()

class OrderItem(BaseModel):
    name: str = Field(description="Название товара (пицца, напиток и т.д.)")
    size: str = Field(description="Размер или объем (например, 25 см, 0.5 л)")
    quantity: int = Field(description="Количество единиц товара")
    price: int = Field(description="Цена за одну единицу (из контекста)")

class OrderState(BaseModel):
    items: List[OrderItem] = []
    total_price: int = Field(description="Общая сумма заказа", default=0)
    message_to_user: str = Field(description="Ответ пользователю (подтверждение или вопрос)")
    
system_prompt = (
    "Ты — оператор пиццерии. Твоя задача — формировать и обновлять заказ пользователя. "
    "У тебя есть: \n1. КОНТЕКСТ МЕНЮ: {context}\n2. ТЕКУЩИЙ ЗАКАЗ: {current_order}\n"
    "Инструкции:\n"
    "- Если пользователь просит добавить товар, найди цену в меню и добавь в список.\n"
    "- Если пользователь просит изменить/удалить, обнови ТЕКУЩИЙ ЗАКАЗ.\n"
    "- Всегда пересчитывай total_price.\n"
    "- Если товара нет в меню, вежливо скажи об этом в message_to_user."
)

openai_api_key=os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in environment variables.")

embeddings = OpenAIEmbeddings(#api_key=SecretStr(openai_api_key), 
                              model='text-embedding-3-small', 
                              base_url="https://api.vsellm.ru/")
db = FAISS.load_local("data/dodo_faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever()

llm = ChatOpenAI(#api_key=SecretStr(openai_api_key), 
                 model='gpt-4o-mini',
                 base_url="https://api.vsellm.ru/")  

prompt = PromptTemplate(
    input_variables=["input", "context", "current_order", "chat_history"],
    template=system_prompt + "\n\nПользователь: {input}\nТекущий заказ: {current_order}\nКонтекст меню: {context}\nИстория чата: {chat_history}."
)
chain = prompt | llm.with_structured_output(OrderState)

# Загружаем модель Whisper локально (она отлично работает на Arch)
# stt_model = whisper.load_model("base")