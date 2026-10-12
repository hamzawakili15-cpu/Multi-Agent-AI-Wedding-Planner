from tavily import TavilyClient
from langchain.tools import tool
from dotenv import load_dotenv
import logging
load_dotenv()


# configuration for logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

Tavily_Client=TavilyClient()
@tool 
def web_search(query:str) -> str:
    """Search the web for a given topic and return the results."""
    logging.info(f"Searching the web for topic: {query}")
    return Tavily_Client.search(query)


