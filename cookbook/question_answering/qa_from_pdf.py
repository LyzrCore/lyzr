from pprint import pprint

import openai

from lyzr import QABot

openai.api_key = "sk-"

path = ""

qa_bot = QABot.pdf_qa(
    input_files=[path],
    llm_params={"model": "gpt-3.5-turbo"},
)
_query = ""
res = qa_bot.query(_query)
print(res.response)
