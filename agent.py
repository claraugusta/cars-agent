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
from rag_query import *

load_dotenv()

print("Iniciando o teste...")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def montar_contexto(resultados):
    ctx_linhas = []
    for r in resultados:
        cite = f"[{r['rank']}] ({r['fonte']} #{r['idx_local']})"
        ctx_linhas.append(f"{cite}\n{r['texto']}")
    return "\n\n".join(ctx_linhas)

def montar_prompt(resultados):
    context = montar_contexto(resultados)
    prompt = f"""
                You are a used car sales assistant who responds based ONLY on the CONTEXT provided.
                If the answer is not in context, say "I couldn't find the information in the documents."

                CONTEXT (excerpts retrieved):
                {context}

                Instructions:
                - Answer in a salesy but short way.
                - If there is conflicting data, point out the discrepancy.
                """
    return prompt.strip()

class AgentState(TypedDict):
    messages: Annotated[list, add_messages] 
    json_result: Optional[str] 
    cars_to_describe: Optional[list] 


class Car_name(BaseModel):
    Make: Optional[str] = Field(description="A car's manufacturer")
    Model: Optional[str] = Field(description="A car's model")

def assistant(state: AgentState):
    last_message_content = state["messages"][-1].content
    hits = recuperar(last_message_content, k=5)
    text_prompt = montar_prompt(hits)
    response = llm.invoke([SystemMessage(content=text_prompt)] + state["messages"])
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
builder.add_node("describe_car", describe_car)

builder.add_edge(START, "assistant")
builder.add_edge("assistant", END)
graph = builder.compile()


user_query = "Existe algum carro Hyundai Santa fe vendido?"
messages_input = [HumanMessage(content=user_query)]

# 2. Invoque o grafo com a entrada no formato correto
# O formato é um dicionário onde a chave corresponde ao estado do grafo ("messages")
final_state = graph.invoke({"messages": messages_input})

# 3. Imprima a resposta final do assistente
# A resposta é a última mensagem na lista de mensagens do estado final
assistant_response = final_state["messages"][-1]

print("\n--- Resposta do Assistente ---")
print(assistant_response.content)
