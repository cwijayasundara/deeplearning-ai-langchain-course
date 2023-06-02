import os
import openai
import warnings

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

warnings.filterwarnings('ignore')

from langchain.agents import load_tools, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType

llm = ChatOpenAI(temperature=0.0)

tools = load_tools(["ddg-search", "wikipedia", "python_repl"])

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=False)

try:
    result = agent("who won the 2022 world cup")
    print(result)
except:
    print("exception on external access")

question = "Tom M. Mitchell is an American computer scientist \
and the Founders University Professor at Carnegie Mellon University (CMU)\
what book did he write?"

try:
    result = agent(question)
    print(result)
except:
    print("exception on external access")

# Use the Python REPL

customer_list = [["Zak", "Smith"], ["Kit", "Doe"], ["Albert", "Doe"], ["Paul", "Walker"]]

result = agent.run(
    f""" Sort these customers by last name first and then first name and then print the result: {customer_list} """)
print(result)

# Create your own tool

from langchain.agents import tool
from datetime import date


@tool
def time(text: str) -> str:
    """Returns todays date, use this for any \
    questions related to knowing todays date. \
    The input should always be an empty string, \
    and this function will always return todays \
    date - any date mathmatics should occur \
    outside this function."""
    return str(date.today())

agent = initialize_agent(
    tools + [time],
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=False)

try:
    result = agent("whats the date today?")
    # I want to print the output from the agent
    print(result)
except:
    print("exception on external access")
