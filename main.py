import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
import services.rag.legal_rag as legal_rag
from services.rag.legal_rag import constitution_tool, law_search_tool

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Initialize LLM
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
)

# ------------------ Initialize Agents ------------------
researcher_tools = [law_search_tool]
advisor_tools = [constitution_tool]

researcher_agent = initialize_agent(
    tools=researcher_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

advisor_agent = initialize_agent(
    tools=advisor_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# ------------------ Request Schema ------------------
class LegalQueryRequest(BaseModel):
    query: str

# ------------------ Endpoint ------------------
@app.post("/legal-query")
def handle_legal_query(request: LegalQueryRequest):
    query = request.query
    print(f"\n🔍 Researcher handling query: {query}")
    response = researcher_agent.run(query)

    return {
        "researcher_result": response,
        "latest_law_results": legal_rag.latest_law_results,
    }
