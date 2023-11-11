from pprint import pprint

import openai

from lyzr import ChatBot

openai.api_key = "sk-"

path = ""

rag = ChatBot.docx_chat(
    input_files=[path],
    llm_params={"model": "gpt-3.5-turbo"},
)

_query = ""

rag = rag.chat(_query)

pprint(rag.response)
