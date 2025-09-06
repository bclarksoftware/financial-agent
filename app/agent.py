import operator
from typing import TypedDict, Annotated, Sequence

from dotenv import load_dotenv
from langchain_community.tools import PolygonLastQuote, PolygonTickerNews, PolygonFinancials, PolygonAggregates
from langchain_community.utilities.polygon import PolygonAPIWrapper
from langchain_core.messages import BaseMessage
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt.tool_node import ToolNode

from app.tools import discounted_cash_flow, owner_earnings, roic, roe

# Load the environment variables
load_dotenv()

# Choose the LLM that will drive the agent
model = ChatOpenAI(model="gpt-4-0125-preview", streaming=True)

# Create the tools
polygon = PolygonAPIWrapper()
integration_tools = [
    PolygonLastQuote(api_wrapper=polygon),
    PolygonTickerNews(api_wrapper=polygon),
    PolygonFinancials(api_wrapper=polygon),
    PolygonAggregates(api_wrapper=polygon),
]

local_tools = [discounted_cash_flow, roe, roic, owner_earnings]
tools = integration_tools + local_tools

tool_node = ToolNode(tools)

model = model.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state['messages']
    last_message = messages[-1]
    # If there are no tool calls, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there are, we continue
    else:
        return "continue"


# Define the function that calls the model
def call_model(state):
    messages = state['messages']
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define the function to execute tools
def call_tool(state):
    # The ToolNode will automatically execute tools based on tool_calls in the last message
    return tool_node.invoke(state)


# Define a new graph
workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # END is a special node marking that the graph should finish.
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Otherwise we finish.
        "end": END
    }
)

# We now add a normal edge from `tools` to `agent`.
workflow.add_edge('action', 'agent')

# Finally, we compile it!
agent = workflow.compile()
