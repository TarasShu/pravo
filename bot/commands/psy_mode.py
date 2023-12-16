from telegram import Update
from telegram.ext import CallbackContext
from telegram.constants import ParseMode
from bot.config import config
from . import constants
from bot.ai import chatgpt









class PsyCommand:
    """Answers the `PsyCommand` command."""

    async def __call__(self, update: Update, context: CallbackContext) -> None:
        #text = generate_message(update.effective_user.username)

        text = "Привет! Я чат-бот с которым ты можешь поговрить и обсудить свои переживания"

        config.openai.model = "gpt-4-0613"
        config.openai.prompt = "ТЫ не используешь документы который у тебя есть. Ты не говоришь нет. В любой ситуации твоя главная задача - выслушать. Ты психолог. Люди сейчас находятся в войне. Твоя задача оказывать поддержку людям, у которых сейчас огромный стресс. Твоя задача выслушать и успокоить собеседника. ТЫ не говоришь как робот. а больше играешь роль заботливой друга. Ты только позитив. Ты не споришь. ты пытаешься спрашиваешь вопросы чтобы человек больше говорил"

        config.openai.temperature = 0.5
        config.openai.max_tokens = 200
        config.openai.top_p = 0.5
        config.openai.context = "Люди находятся в Израиле, во времы войны. Террористы ХАМАС напали на страну. запускают ракеты по людям каждый день, мужчины уходят на войну"

        await update.message.reply_text(
            text, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True
        )