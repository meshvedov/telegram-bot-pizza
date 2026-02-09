import os
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
import whisper # Локальный STT

# Твои импорты LangChain и Pydantic (OrderState, chain, retriever)
# ...
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
#-----------------------------------------------------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Загружаем модель Whisper локально (она отлично работает на Arch)
stt_model = whisper.load_model("base")

# Словарь для хранения корзин пользователей (вместо st.session_state)
user_carts = {}

@dp.message(Command("start"))
async def start_cmd(message: types.Message):
    user_carts[message.from_user.id] = OrderState(items=[], total_price=0, message_to_user="")
    await message.answer("Привет! Я бот-пиццерия. Можешь писать текстом или прислать голосовое!")

# ОБРАБОТКА ГОЛОСА
@dp.message(F.voice)
async def handle_voice(message: types.Message):
    # 1. Скачиваем файл
    file_id = message.voice.file_id
    file = await bot.get_file(file_id)
    file_path = f"{file_id}.ogg"
    await bot.download_file(file.file_path, file_path)

    # 2. Транскрибация (Whisper)
    result = stt_model.transcribe(file_path)
    user_text = result['text']
    os.remove(file_path) # Чистим за собой

    # 3. Отправляем в твою логику заказа
    await process_order_logic(message, user_text)

# ОБРАБОТКА ТЕКСТА
@dp.message(F.text)
async def handle_text(message: types.Message):
    await process_order_logic(message, message.text)

async def process_order_logic(message: types.Message, user_text: str):
    user_id = message.from_user.id
    
    # Достаем или создаем корзину
    current_cart = user_carts.get(user_id, OrderState(items=[], total_price=0))
    
    # Твоя RAG логика
    context_docs = retriever.invoke(user_text)
    context_text = "\n".join([d.page_content for d in context_docs])
    
    # Вызов Qwen3
    new_state = chain.invoke({
        "input": user_text,
        "context": context_text,
        "current_order": current_cart.model_dump_json(),
        "chat_history": [] # Можно добавить историю из БД
    })

    # Сохраняем состояние
    user_carts[user_id] = new_state

    # Формируем красивый ответ
    cart_msg = "\n".join([f"• {i.name} ({i.size}) x{i.quantity}" for i in new_state.items])
    full_response = (
        f"{new_state.message_to_user}\n\n"
        f"🛒 **Текущая корзина:**\n{cart_msg}\n"
        f"💰 **Итого: {new_state.total_price} ₽**"
    )
    
    await message.answer(full_response, parse_mode="Markdown")

if __name__ == "__main__":
    import asyncio
    asyncio.run(dp.start_polling(bot))
