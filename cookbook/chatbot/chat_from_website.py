from pprint import pprint

import openai

from lyzr import ChatBot

openai.api_key = "sk-"

link = "https://www.nelsongp.com/"

rag = ChatBot.website_chat(
    url=link,
    llm_params={"model": "gpt-3.5-turbo"},
)

_query = "what does nelson do?"

rag = rag.chat(_query)

pprint(rag.response)
