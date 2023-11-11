import openai
from lyzr import ChatBot
from pprint import pprint

openai.api_key = "sk-"

link = ["https://www.youtube.com/watch?v=fcfVjd_oV1I"]

chatbot = ChatBot.youtube_chat(
    urls=link,
    llm_params={"model": "gpt-3.5-turbo"},
)

_query = "what does googler do?"

response = chatbot.chat(_query)

pprint(response.response)

_query = "what did i asked above?"
response = chatbot.chat(_query)
pprint(response.response)


#%%
