from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from mem0 import Memory


def log(message):
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {message}\n"

    with open(".log", 'a') as file:
        file.write(entry)


load_dotenv()

local_llm = ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        #temperature=0.7,
)

langchain_embeddings = HuggingFaceEmbeddings(
    model_name="sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja",
    model_kwargs={'device': 'cpu'}
)

vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=langchain_embeddings,
    collection_name="mem0"
)

config = {
    "vector_store": {
        "provider": "langchain",
        "config": {
            "client": vector_store
        }
    },
    "llm": {
        "provider": "langchain",
        "config": {
            "model": local_llm
        }
    },
        "embedder": {
        "provider": "langchain",
        "config": {
            "model": langchain_embeddings,
            "embedding_dims": 1536
        }
    }
}

memory = Memory.from_config(config)
memory.last_mem_id = ""

mcp = FastMCP("Memory")

@mcp.tool()
def remember(search_query: str) -> str:
    """Search for relevant memory from conversation."""
    log(f"Recollect: ({search_query})")
    try:
        results = memory.search(query=search_query, user_id="user")
    except:
        results = "no related memory"

    if not results or results == {'results': []}:
        log("memory: no related memory")
        results = "no related memories, try web search"
    log(f"memory result: {results}")
    return str(results)


@mcp.tool()
def remember_last() -> str:
    """Retrieve previous exchange."""
    log(f"Recollect: previous exchange.)")
    try:
        if memory.last_mem_id:
            results = memory.get(memory.last_mem_id)
        else:
            results = "no previous exchange"
    except:
            results = "no previous exchange"

    if results == []:
            log("memory: no previous exchange")
            results = "no previous exchange"
    return str(results)

if __name__ == "__main__":
    mcp.run(transport="stdio")
