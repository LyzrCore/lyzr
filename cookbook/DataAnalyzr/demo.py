from lyzr import DataAnalyzr
import pandas as pd
import openai

openai.api_key = "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"

df = pd.read_csv("./cars_prices.csv")
user_query = "what is the impact of milage on the price of 2wd cars"

da = DataAnalyzr(df, user_query)
model_response = da.getCode()
exec(model_response)