from pprint import pprint

import openai

from lyzr import ChatBot

openai.api_key = "sk-"

import nest_asyncio

nest_asyncio.apply()

link = "https://archlinux.org/"
rag = ChatBot.webpage_chat(
    url=link,
    llm_params={"model": "gpt-3.5-turbo"},
    embed_model=""
)

_query = "what does akinobu do?"

rag = rag.chat(_query)

pprint(rag.response)

#%%
