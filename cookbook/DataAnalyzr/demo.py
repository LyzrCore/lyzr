from lyzr import DataAnalyzr
import pandas as pd
import openai

openai.api_key = "sk-"

df = pd.read_csv("./cars_prices.csv")
user_query = "what is the impact of milage on the price of 2wd cars"

da = DataAnalyzr(df, user_query)
model_response = da.getCode()
exec(model_response)
