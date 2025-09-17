from dotenv import load_dotenv

# Importa o modelo de chat da OpenAI e uma mensagem
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, END, START
from trustcall import create_extractor
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Literal, Optional, TypedDict, Annotated
from select_cars import select_cars
import json
import random

load_dotenv()

print("Iniciando o teste...")

# Inicializa o modelo
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class AgentState(TypedDict):
    messages: Annotated[list, add_messages] 
    json_result: Optional[str] 
    cars_to_describe: Optional[list] 


class Car_name(BaseModel):
    Manufacturer: Optional[str] = Field(description="A car's manufacturer")
    Model: Optional[str] = Field(description="A car's model")

@tool(args_schema=Car_name)
def car_search_tool(Manufacturer: Optional[str] = None, Model: Optional[str] = None):
    """Search the vehicle database by manufacturer and/or model.
    Use this tool whenever a user asks for vehicle information.
    """
    json_result = select_cars('Manufacturer', Manufacturer, 'Model', Model)
    return json_result

tools = [car_search_tool]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools, tool_choice="car_search_tool" )

def agent(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return{"messages": [response]}

def process_json(state: AgentState):
    """Gets the JSON from the ToolMessage, picks 3 cars and updates the state."""
    tool_message = state["messages"][-1]
    
    json_string = tool_message.content
    car_list = json.loads(json_string)  
    random_cars = random.sample(car_list, max(3, len(car_list))) 
    return {"cars_to_describe": random_cars}

def describe_car(state: AgentState):
    """Describe the 3 random cars based on the json informations"""
    cars_to_describe = state["cars_to_describe"]
    if not cars_to_describe:
        text_prompt = "The user requested information, but the search returned no results. Please advise them."
    else:
        text_prompt = f"""
        The user requested information about cars. The search returned the following vehicles:
        {cars_to_describe}

        Your task is to write a short, friendly, and salesy "prompt" for each of these cars,
        using the information in the JSON.
        """
    response = llm.invoke([SystemMessage(content=text_prompt)] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    """Decide para onde ir depois do nó 'agent'."""
    print("--- Nó: Condicional (Roteador) ---")
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        print("Decisão: Chamar a ferramenta.")
        return "call_tool" 
    else:
        return END
    

builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("describe_car",describe_car)
builder.add_node("action", tool_node)
builder.add_node("process_json",process_json)
builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "call_tool": "action",
        END: END
    })
builder.add_edge("action", "process_json")
builder.add_edge("process_json", "describe_car")
builder.add_edge("describe_car", END)

graph = builder.compile()
input_messages = [HumanMessage(content="ford")]
config = {"configurable": {"thread_id": "1"}}

for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()