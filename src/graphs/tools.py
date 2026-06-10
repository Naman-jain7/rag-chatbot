from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import requests
from app.core.config import settings

search_tool = DuckDuckGoSearchRun()


@tool
def calculator(num1: float, num2: float, opr: str) -> dict:
    """
    performs basic arithmetic operations on 2 numbers
    supported operations: add, sub, mul, div
    """
    try:
        if opr == "add":
            result = num1 + num2
        elif opr == "sub":
            result = num1 - num2
        elif opr == "mul":
            result = num1 * num2
        elif opr == "div":
            if num2 == 0:
                return {"error": "Division by zero is not allowed"}
            result = num1 / num2
        else:
            return {"error": f"Unsupported opr '{opr}'"}

        return {
            "first_num": num1,
            "second_num": num2,
            "operation": opr,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={settings.services.ALPHAVANTAGE_STOCK_API_KEY}"
    r = requests.get(url)
    return r.json()


@tool
def get_currency_exchange_rate(from_currency: str, to_currency: str) -> dict:
    """
    Fetch latest currency exchange rate for a given currency pair (e.g. 'USD', 'EUR')
    using ExchangeRate-API with API key in the URL.
    """
    url = f"https://v6.exchangerate-api.com/v6/{settings.services.CURRENCY_EXCHANGE_API_KEY}/latest/{from_currency}"
    r = requests.get(url)
    r = r.json()
    return {"conversion_rate": r["conversion_rate"]}


tools = [search_tool, calculator, get_stock_price, get_currency_exchange_rate]
