from telegram import Update
from telegram.ext import CallbackContext
from telegram.constants import ParseMode
from bot.config import config
from . import constants
from bot.ai import chatgpt


class RegCommand:
    """Answers the `help` command."""

    async def __call__(self, update: Update, context: CallbackContext) -> None:
        #text = generate_message(update.effective_user.username)
        text = "Привет! Я чат-бот, с которым ты можешь поговорить и обсудить свои переживания. Как я могу тебе помочь?"
        config.openai.model = config.openai.default_model
        config.openai.prompt = config.openai.default_prompt






        await update.message.reply_text(
            text, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True
        )

   

            
   