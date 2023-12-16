"""DALL-E model from OpenAI."""


from bot.config import config


from openai import AsyncOpenAI

aclient = AsyncOpenAI(api_key="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")#config.openai.api_key)

class Model:
    """OpenAI DALL-E wrapper."""

    async def imagine(self, prompt: str, size: str) -> str:
        """Generates an image of the specified size according to the description."""
        resp = await aclient.images.generate(prompt=prompt, size=size, n=1)
        if len(resp.data) == 0:
            raise ValueError("received an empty answer")
        return resp.data[0].url
