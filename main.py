from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import CommandStart
from dotenv import load_dotenv
load_dotenv()

import os
import asyncio
import logging

bot = Bot(token=os.getenv("BOT_TOKEN"))
dp = Dispatcher()

@dp.message(CommandStart())
async def handle_start(message: types.Message):
    await message.answer(text=f"Привет! Я {message.from_user.full_name}")

@dp.message()
async def echo_message(message: types.Message):
    await bot.send_message(
        chat_id=message.chat.id,
        text='Wait a second...'
    )
    await message.answer(text=message.text)
    await message.reply(text=message.text)

async def main():
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
