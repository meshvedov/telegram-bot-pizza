from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import CommandStart
from dotenv import load_dotenv

import os
import asyncio
import logging

from app.handlers import router, bot

async def main():
    dp = Dispatcher()
    dp.include_router(router)
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
