import streamlit as st
from langchain_openai import ChatOpenAI
# Импортируй свои ранее созданные OrderState, retriever и prompt здесь
from typing import List, Optional
from pydantic import BaseModel, Field, SecretStr
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv
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

st.set_page_config(page_title="Pizza Order Bot", layout="centered")
st.title("🍕 Пиццерия Qwen")

# 1. Инициализация состояния (выполняется один раз)
if "current_cart" not in st.session_state:
    st.session_state.current_cart = OrderState(items=[], total_price=0, message_to_user="Привет! Что желаете заказать?")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Подключение к Qwen (LM Studio)
# llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio", model="qwen3-30b")
openai_api_key=os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in environment variables.")

embeddings = OpenAIEmbeddings(#api_key=SecretStr(openai_api_key), 
                              model='text-embedding-3-small', 
                              base_url="https://api.vsellm.ru/")
db = FAISS.load_local("notebooks/dodo_faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever()

llm = ChatOpenAI(#api_key=SecretStr(openai_api_key), 
                 model='gpt-4o-mini',
                 base_url="https://api.vsellm.ru/")  

prompt = PromptTemplate(
    input_variables=["input", "context", "current_order", "chat_history"],
    template=system_prompt + "\n\nПользователь: {input}\nТекущий заказ: {current_order}\nКонтекст меню: {context}\nИстория чата: {chat_history}."
)
chain = prompt | llm.with_structured_output(OrderState)

def clear_chat():
    st.session_state.chat_history = []
    st.session_state.current_cart = OrderState(items=[], total_price=0, message_to_user="Привет! Что желаете заказать?")

# 2. Боковая панель с корзиной (визуализация состояния)
with st.sidebar:
    st.header("🛒 Ваша корзина")
    if not st.session_state.current_cart.items:
        st.write("Корзина пуста")
    else:
        for item in st.session_state.current_cart.items:
            st.write(f"**{item.name}** ({item.size})")
            st.write(f"{item.quantity} x {item.price} ₽")
            st.divider()
        st.subheader(f"Итого: {st.session_state.current_cart.total_price} ₽")
        if st.button("Оформить заказ", on_click=clear_chat):
            st.success("Заказ отправлен на кухню!")

# 3. Отображение истории чата
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

# 4. Поле ввода
if user_input := st.chat_input("Напишите ваш заказ..."):
    # Добавляем вопрос в чат
    st.session_state.chat_history.append(("human", user_input))
    with st.chat_message("human"):
        st.markdown(user_input)

    # Логика обработки (RAG + LLM)
    with st.chat_message("ai"):
        with st.spinner("Считаю..."):
            # Поиск в RAG
            context_docs = retriever.invoke(user_input)
            context_text = "\n".join([d.page_content for d in context_docs])
            
            # Вызов модели
            new_state = chain.invoke({
                "input": user_input,
                "context": context_text,
                "current_order": st.session_state.current_cart.model_dump_json(),
                "chat_history": st.session_state.chat_history[-6:] # Окно памяти
            })

            # Обновляем состояние
            st.session_state.current_cart = new_state
            st.session_state.chat_history.append(("ai", new_state.message_to_user))
            st.markdown(new_state.message_to_user)
            
            # Принудительно обновляем интерфейс, чтобы корзина в сайдбаре обновилась
            st.rerun()
