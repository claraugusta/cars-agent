from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, END, START
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional, TypedDict, Annotated
from select_cars import select_cars
import json
import random

load_dotenv()

print("Iniciando o teste...")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

class AgentState(TypedDict):
    messages: Annotated[list, add_messages] 
    json_result: Optional[str] 
    cars_to_describe: Optional[list] 


class Car_name(BaseModel):
    Make: Optional[str] = Field(description="A car's manufacturer")
    Model: Optional[str] = Field(description="A car's model")

@tool(args_schema=Car_name)
def car_search_tool(Make: Optional[str] = None, Model: Optional[str] = None):
    """If the user mentioned a car, search the vehicle database by manufacturer and/or model.
    Use this tool whenever a user asks for vehicle information.
    """
    json_result = select_cars('make', Make, 'model', Model)
    list_of_cars = json.loads(json_result)
    random_cars = random.sample(list_of_cars, min(3, len(list_of_cars))) 
    return {"cars_to_describe": random_cars}

tools = [car_search_tool]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)

def assistant(state: AgentState):
    text_prompt = '''You are a used car sales assistant. If the user asks a question that is not related to searching for 
    vehicle information, politely inform them that you can only help with questions about cars and ask if they would 
    like to search by model or manufacturer.'''

    response = llm_with_tools.invoke([SystemMessage(content=text_prompt)] + state["messages"])
    return {"messages": [response]}


def describe_car(state: AgentState):
    """Describe the cars based on the json informations"""
    cars_to_describe = state.get('cars_to_describe')
    print(cars_to_describe)
    text_prompt = f"""The user requested information about cars. The search returned the following vehicles:
    {cars_to_describe}

    Your task is to write a short, friendly, and salesy "prompt" for each of these used cars,
    using the information in the JSON. In the end, recommend the best purchase option among the cars.
    """
    response = llm.invoke([SystemMessage(content=text_prompt)] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    """Decide where to go after the 'assistant' node."""
    print("--- Node: Conditional (Router) ---")
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        print("Decision: Call the tool.")
        return "call_tool"
    else:
        print("Decision: No tool called, respond politely.")
        return END

builder = StateGraph(MessagesState)

builder.add_node("assistant", assistant)
builder.add_node("car_search_tool", tool_node)
builder.add_node("describe_car", describe_car)

builder.add_edge(START, "assistant")

builder.add_conditional_edges(
"assistant",
should_continue,
    {"call_tool": "car_search_tool",
    END:END,
}
)
builder.add_edge("car_search_tool", "describe_car")
builder.add_edge("describe_car", END)
graph = builder.compile()

