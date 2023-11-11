from pprint import pprint

import openai

from lyzr import QABot

openai.api_key = "sk-"

link = "https://www.nelsongp.com/"

rag = QABot.website_qa(
    url=link,
    llm_params={"model": "gpt-3.5-turbo"},
)

_query = "what does nelson do?"

rag = rag.query(_query)

pprint(rag.response)
