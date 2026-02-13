import os
from dotenv import load_dotenv
from aiogram import Router, F, types, Bot
from aiogram.types import Message
from aiogram.filters import Command, CommandStart
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext

from app.generators import retriever, chain, OrderState, stt_model

load_dotenv()
router = Router()
bot = Bot(token=os.getenv("BOT_TOKEN"))

class Gen(StatesGroup):
    waiting_for_input = State()

user_carts = {} # хранилище для заказов

@router.message(Command("start"))
async def start_cmd(message: Message):
    user_carts[message.from_user.id] = OrderState(items=[], total_price=0, message_to_user="")
    await message.answer("Привет! Я бот-пиццерия. Можешь писать текстом или прислать голосовое!")
    
@router.message(F.text == "Оформить заказ")
async def handle_order(message: Message):
    # проверка наличия заказа
    current_cart = user_carts.get(message.from_user.id)
    if not current_cart or not current_cart.items:
        await message.answer("Ваша корзина пуста. Пожалуйста, добавьте товары перед оформлением заказа.")
        return
    # всплывающее сообщение с подтверждением
    await message.answer("Ваш заказ оформлен! Спасибо за покупку!")
    # очищаем корзину
    user_carts[message.from_user.id] = OrderState(items=[], total_price=0, message_to_user="")
    
@router.message(F.text == "Очистить корзину")
async def handle_clear_cart(message: Message):
    user_carts[message.from_user.id] = OrderState(items=[], total_price=0, message_to_user="Корзина очищена. Что желаете заказать?")
    await message.answer("Корзина очищена. Что желаете заказать?") 
    
@router.message(Gen.waiting_for_input)
async def stop_flood(message: Message):
    await message.answer("Пожалуйста, подождите, я обрабатываю ваш предыдущий запрос.")

# ОБРАБОТКА ГОЛОСА
@router.message(F.voice)
async def handle_voice(message: Message, state: FSMContext):
    # import pdb; pdb.set_trace()
    await state.set_state(Gen.waiting_for_input)
    # 1. Скачиваем файл
    file_id = message.voice.file_id
    file = await bot.get_file(file_id)
    file_path = f"{file_id}.ogg"
    await bot.download_file(file.file_path, file_path)

    # 2. Транскрибация (Whisper)
    menu_context = "Пицца, Пепперони, Маргарита, сок Добрый, кола, газировка, четыре сыра, 25 см, 30 см."
    result = stt_model.transcribe(file_path, language='ru', initial_prompt=menu_context)
    # user_text = result['text']
    segments, info = stt_model.transcribe(file_path, beam_size=5, initial_prompt=menu_context, language='ru')
    user_text = "".join([segment.text for segment in segments])
    print(user_text)
    os.remove(file_path) # Чистим за собой

    # 3. Отправляем в твою логику заказа
    await process_order_logic(message, user_text)
    await state.clear()

# ОБРАБОТКА ТЕКСТА
@router.message(F.text)
async def handle_text(message: Message, state: FSMContext):
    await state.set_state(Gen.waiting_for_input)
    await process_order_logic(message, message.text)
    await state.clear()

async def process_order_logic(message: Message, user_text: str):
    # import pdb; pdb.set_trace()
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
    
    await message.answer(full_response, parse_mode="Markdown", reply_markup=create_reply_keyboard())
    
def create_reply_keyboard() -> types.ReplyKeyboardMarkup:
    buttons = [
        [types.KeyboardButton(text="Оформить заказ"), types.KeyboardButton(text="Очистить корзину")],
    ]
    return types.ReplyKeyboardMarkup(keyboard=buttons, resize_keyboard=True, is_persistent=True)
    
