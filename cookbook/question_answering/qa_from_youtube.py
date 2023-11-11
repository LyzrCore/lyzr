from lyzr import QABot

link = ["https://www.youtube.com/watch?v=fcfVjd_oV1I"]

rag = QABot.youtube_qa(
    urls=link,
    llm_params={"model": "gpt-3.5-turbo"},
)

_query = "what does googler do?"

response = rag.query(_query)


_query = "what did i asked above?"
response = rag.query(_query)
print(response)