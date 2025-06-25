import os
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.memory import VectorStoreRetrieverMemory
from services.rag.legal_rag import embeddings
from langchain_community.vectorstores import Chroma
import services.rag.legal_rag as legal_rag
from services.rag.legal_rag import constitution_tool, law_search_tool
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Legal Research Agent API",
    description="An API for interacting with a legal research agent with persistent chat sessions.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LLM
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
)

# ------------------ Agent and Chat Management ------------------
chat_agents: Dict[str, AgentExecutor] = {}
chat_histories: Dict[str, List[Dict[str, str]]] = {}


# ------------------ Request Schema ------------------
class LegalQueryRequest(BaseModel):
    query: str


# ------------------ Endpoints ------------------

@app.post("/chat", summary="Create a new chat session")
def create_chat():
    """
    Initializes a new chat session with its own memory.
    """
    chat_id = str(uuid.uuid4())
    researcher_tools = [law_search_tool]

    vectorstore = Chroma(
        collection_name=f"chat_memory_{chat_id}",
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs=dict(k=5))

    memory = VectorStoreRetrieverMemory(retriever=retriever, memory_key="history", return_messages=False)

    researcher_agent = initialize_agent(
        tools=researcher_tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
    )

    chat_agents[chat_id] = researcher_agent
    chat_histories[chat_id] = []  # ADDED: Initialize an empty list for the history
    print(f"✅ New chat created: {chat_id}")
    return {"chat_id": chat_id}


@app.get("/chats", summary="List all active chat sessions")
def list_chats():
    """
    Returns a list of all active chat_ids.
    """
    return {"chat_ids": list(chat_agents.keys())}


@app.get("/chat/{chat_id}/history", summary="Get the conversation history for a chat session")
def get_chat_history(chat_id: str):
    if chat_id not in chat_histories:
        raise HTTPException(status_code=404, detail="Chat session not found")

    # Return the history from our simple list
    return {"history": chat_histories[chat_id]}


@app.delete("/chat/{chat_id}", summary="Delete a chat session")
def delete_chat(chat_id: str):
    """
    Deletes a specific chat session and its associated memory.
    """
    if chat_id not in chat_agents:
        raise HTTPException(status_code=404, detail="Chat session not found")

    del chat_agents[chat_id]
    if chat_id in chat_histories:
        del chat_histories[chat_id]
    print(f"🗑️ Chat deleted: {chat_id}")
    return {"status": "success", "message": f"Chat {chat_id} deleted."}


@app.post("/chat/{chat_id}/query", summary="Post a query to a chat session")
def handle_legal_query(chat_id: str, request: LegalQueryRequest):
    """
    Handles a legal query within the context of a specific chat session.
    The agent will have memory of previous interactions in this same chat.
    """
    if chat_id not in chat_agents:
        raise HTTPException(status_code=404, detail="Chat session not found")

    researcher_agent = chat_agents[chat_id]
    query = request.query

    print(f"\n🔍 Researcher handling query for chat {chat_id}: {query}")

    response = researcher_agent.invoke({"input": query})

    ai_response = response.get("output", "No response from agent.")
    chat_histories[chat_id].append({"type": "human", "content": query})
    chat_histories[chat_id].append({"type": "ai", "content": ai_response})

    related_laws = constitution_tool(ai_response)

    return {
        "researcher_result": ai_response,
        "related_cases": legal_rag.related_cases,
        "relevant_laws": related_laws,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")