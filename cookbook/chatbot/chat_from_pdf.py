from pprint import pprint

import openai
from lyzr import ChatBot

openai.api_key = "sk-"
path = ""

qa_bot = ChatBot.pdf_chat(
    input_files=[path],
    llm_params={"model": "gpt-3.5-turbo"},
)
_query = ""
res = qa_bot.chat(_query)
print(res.response)
