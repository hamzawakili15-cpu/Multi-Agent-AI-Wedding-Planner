from langchain.agents import create_agent
from langchain.tools import tool
from models import model
from langchain.messages import HumanMessage
from tools import web_search
import logging

# configuration for logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# Create the subagents

logging.info("initialize the Subagent1...")
subagent1 = create_agent(model=model, tools=[web_search], name="Subagent1")

logging.info("initialize the Subagent2...")
subagent2 = create_agent(model=model, tools=[web_search], name="Subagent2")

@tool
def delegate_subagent1(query:str) ->str:
    """search the web on a given topic by delegating the task to Subagent1 to do so."""
    logging.info(f"delegating query to Subagent1: {query}")
    responce = subagent1.invoke({"messages": [HumanMessage(content=query)]})
    return responce["messages"][-1].content

@tool
def delegate_subagent2(query:str) ->str:
    """search the web on a given topic by delegating the task to Subagent2 to do so."""
    logging.info(f"delegating query to Subagent2: {query}")
    responce = subagent2.invoke({"messages": [HumanMessage(content=query)]})
    return responce["messages"][-1].content


