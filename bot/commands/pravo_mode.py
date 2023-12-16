from telegram import Update
from telegram.ext import CallbackContext
from telegram.constants import ParseMode
from bot.config import config
from . import constants
from bot.ai import chatgpt




class PravoCommand:
    """Answers the `PsyCommand` command."""

    async def __call__(self, update: Update, context: CallbackContext) -> None:
        #text = generate_message(update.effective_user.username)

        text = "Привет! Я чат-бот юрист"       

        await update.message.reply_text(
            text, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True
        )

