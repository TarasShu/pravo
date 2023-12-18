"""ChatGPT (GPT-3.5+) language model from OpenAI."""

import logging
from sys import exc_info

from openai import OpenAI
from openai import AsyncOpenAI
from dotenv import load_dotenv


import os


import tiktoken
from bot.config import config

#from trans import Translator

#from langchain.vectorstores import Pinecone
#from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone




#import Translator

logger = logging.getLogger(__name__)




load_dotenv()


api_openai = os.getenv('OPENAI_API_KEY')


client = OpenAI(
  api_key=api_openai,#os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)

aclient = AsyncOpenAI(api_key=api_openai)



encoding = tiktoken.get_encoding("cl100k_base")
pinecone.init(api_key=os.getenv('PINECONE_API_KEY'),
              environment="gcp-starter")

index = pinecone.Index("pravo")


def _prepare_answer(resp) -> str:
    """Post-processes an answer from the language model."""
    if len(resp.choices) == 0:
        raise ValueError("received an empty answer")

    answer = resp.choices[0].message.content
    answer = answer.strip()

    return answer


def _generate_messages(question: str, history: list[tuple[str, str]]) -> list[dict]:
    """Builds message history to provide context for the language model."""

    contexts = []

    getEmbedding = client.embeddings.create(input=[question], model="text-embedding-ada-002")#config.openai.embedding_model)

    # retrieve from Pinecone
    xq = getEmbedding.data[0].embedding
    # get relevant contexts (including the questions)
    
    getEmbedding = index.query(xq, top_k=5, include_metadata=True)



    contexts = getEmbedding#[item['metadata']['text'] for item in getEmbedding['matches']]
    context = [item['metadata'] for item in contexts['matches']]
    context_strings = [str(item) for item in context]

    augmented_query = "\n\n---\n\n".join(context_strings) + "\n\n-----\n\n" + question

    

    #augmented_query = "\n\n---\n\n".join(contexts) + "\n\n-----\n\n" + question

    messages = [{"role": "system", "content": augmented_query}]

    for prev_question, prev_answer in history:
        messages.append({"role": "user", "content": prev_question})
        messages.append({"role": "assistant", "content": prev_answer})
    messages.append({"role": "user", "content": question})

    

    return messages


def _prepare_params() -> dict:
    params = config.openai.params.copy()
    if config.openai.azure:
        params["api_type"] = "azure"
        params["api_base"] = config.openai.azure["endpoint"]
        params["api_version"] = config.openai.azure["version"]
        params["deployment_id"] = config.openai.azure["deployment"]
    return params


class Model:
    """OpenAI API wrapper."""

    def __init__(self, name: str) -> None:
        """Creates a wrapper for a given OpenAI large language model."""
        self.name = name

    async def ask(self, question: str, history: list[tuple[str, str]]) -> str:
        """Asks the language model a question and returns an answer."""
        # maximum number of input tokens
        n_input = _calc_n_input(self.name, n_output=config.openai.params["max_tokens"])
        messages = _generate_messages(question, history)
        messages = shorten(messages, length=n_input)
        params = _prepare_params()

        resp = await aclient.chat.completions.create(model=self.name,
        messages=messages,
        **params)
        logger.debug(
            "prompt_tokens=%s, completion_tokens=%s, total_tokens=%s",
            resp.usage.prompt_tokens,
            resp.usage.completion_tokens,
            resp.usage.total_tokens,
        )
        answer = _prepare_answer(resp)
        return answer


def shorten(messages: list[dict], length: int) -> list[dict]:
    """
    Truncates messages so that the total number or tokens
    does not exceed the specified length.
    """
    lengths = [len(encoding.encode(m["content"])) for m in messages]
    total_len = sum(lengths)
    if total_len <= length:
        return messages

    # exclude older messages to fit into the desired length
    # can't exclude the prompt though
    prompt_msg, messages = messages[0], messages[1:]
    prompt_len, lengths = lengths[0], lengths[1:]
    while len(messages) > 1 and total_len > length:
        messages = messages[1:]
        first_len, lengths = lengths[0], lengths[1:]
        total_len -= first_len
    messages = [prompt_msg] + messages
    if total_len <= length:
        return messages

    # there is only one message left, and it's still longer than allowed
    # so we have to shorten it
    maxlen = length - prompt_len
    tokens = encoding.encode(messages[1]["content"])
    tokens = tokens[:maxlen]
    messages[1]["content"] = encoding.decode(tokens)


    return messages


def _calc_n_input(name: str, n_output: int) -> int:
    """
    Calculates the maximum number of input tokens
    according to the model and the maximum number of output tokens.
    """
    # OpenAI counts length in tokens, not charactes.
    # We need to leave some tokens reserved for the output.
    n_total = 4096  # max 4096 tokens total by default
    if name == "gpt-4":
        n_total = 8192
    return n_total - n_output
