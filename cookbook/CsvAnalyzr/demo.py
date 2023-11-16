from lyzr import CsvAnalyzr
import openai; openai.api_key = "sk-"

exec(CsvAnalyzr("./cars_prices.csv", "Impact of Milage on Price of 2wd cars").run())