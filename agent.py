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

def montar_prompt(userInput,resultados):
    context = montar_contexto(resultados)
    prompt = f"""
                You are a helpful and friendly used car sales assistant.
                
                First, try to answer the user's question using the CONTEXT provided below.
                If the context helps answer the question, please cite the source using the [n] format.
                
                If the answer is not in the context, use your general knowledge to provide a helpful response.

                USER INPUT:
                {userInput}
                
                CONTEXT (excerpts retrieved):
                {context}
                """
    return prompt.strip()

class AgentState(TypedDict):
    messages: Annotated[list, add_messages] 

def assistant(state: AgentState):
    last_message_content = state["messages"][-1].content
    hits = recuperar(last_message_content, k=5)
    text_prompt = montar_prompt(last_message_content, hits)
    response = llm.invoke([SystemMessage(content=text_prompt)] + state["messages"])
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_edge(START, "assistant")
builder.add_edge("assistant", END)
graph = builder.compile()

user_query = "quero um chevrolet camaro barato, qual voce me recomenda?"
messages_input = [HumanMessage(content=user_query)]

final_state = graph.invoke({"messages": messages_input})
assistant_response = final_state["messages"][-1]

print("\n--- Resposta do Assistente ---")
print(assistant_response.content)