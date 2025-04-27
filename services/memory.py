from langchain_huggingface import HuggingFaceEmbeddings
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

config = {
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


mcp = FastMCP("Memory")

@mcp.tool()
def recollect(search_query: str) -> str:
        """
        Search for memories
        Args:
        query: The search query.
        """
        log(f"Recollect: ({search_query})")

        results = memory.search(query=search_query, user_id="user")
        return str(results)

if __name__ == "__main__":
    mcp.run(transport="stdio")
