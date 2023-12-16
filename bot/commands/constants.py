"""Bot command constants."""

HELP_MESSAGE = """
Контакт для связи @mendoli
"""

PRIVACY_MESSAGE = (
    "⚠️ The bot does not have access to group messages, "
    "so it cannot reply in groups. Use @botfather "
    "to give the bot access (Bot Settings > Group Privacy > Turn off)"
)





BOT_COMMANDS = [
   # ("retry", "retry the last question"),
   # ("imagine", "generate described image"),
    ("version", "show debug information"),
    ("help", "show help"),
    #("reg_mode", "Обычный чат"),
    #("psy_mode", "Психологический чат"),
    #("pravo_mode", "Юридический чат")
]

ADMIN_COMMANDS = {
    "config": "view or edit the config"
}