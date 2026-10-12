from prompts import WEDDING_PLANNER_AGENT_PROMPT, USER_PROMPT_FOR_MAIN_AGENT
from agent import delegate_subagent1, delegate_subagent2
from models import model
from langchain.agents import create_agent
import logging
from langchain.messages import HumanMessage

# configuration for logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# asking for their requirements for their wedding
logging.info('Asking for their requirements...')
user_requirements = input("Please enter your requirements and preference of your wedding:\n")

# Updating the WEDDING_PLANNER_AGENT_PROMPT
logging.info("Updating the wedding planner agent prompt...")
updated_wedding_prompt = WEDDING_PLANNER_AGENT_PROMPT.format(requirements=user_requirements)

#create the main wedding planner agent 
logging.info("initialize the main wedding planner agent... ")

wedding_planner_agent = create_agent(
    model=model,
    tools=[delegate_subagent1, delegate_subagent2],
    system_prompt=updated_wedding_prompt,
    name="WeddingPlannerAgent"
)

# invoke the main agent
logging.info("invoking Main wedding planner agent...")
response = wedding_planner_agent.invoke({"messages": [HumanMessage(content=USER_PROMPT_FOR_MAIN_AGENT)]})

#print the response
print("\nMain Wedding Planner Agent's Response:")
print(response["messages"][-1].content)









