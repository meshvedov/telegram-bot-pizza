import os
from aiogram import Router, F
from aiogram.types import Message
from aiogram.filters import Command, CommandStart

from app.generators import retriever, chain, OrderState

router = Router()

#==============================
# @router.message(CommandStart())
# async def handle_start(message: Message):
#     await message.answer(text=f"Привет! Я {message.from_user.full_name}")

# @router.message(F.text)
# async def echo_message(message: Message):
#     # await bot.send_message(
#     #     chat_id=message.chat.id,
#     #     text='Wait a second...'
#     # )
#     await message.answer(text=message.text)
#     await message.reply(text=message.text)
#==============================

user_carts = {}

@router.message(Command("start"))
async def start_cmd(message: Message):
    user_carts[message.from_user.id] = OrderState(items=[], total_price=0, message_to_user="")
    await message.answer("Привет! Я бот-пиццерия. Можешь писать текстом или прислать голосовое!")

# ОБРАБОТКА ГОЛОСА
@router.message(F.voice)
async def handle_voice(message: Message):
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
@router.message(F.text)
async def handle_text(message: Message):
    await process_order_logic(message, message.text)

async def process_order_logic(message: Message, user_text: str):
    import pdb; pdb.set_trace()
    user_id = message.from_user.id
    
    # Достаем или создаем корзину
    current_cart = user_carts.get(user_id, OrderState(items=[], total_price=0, message_to_user="Привет! Что желаете заказать?"))
    
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
