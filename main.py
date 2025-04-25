from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pprint import pprint
import asyncio
import os

load_dotenv()

#if "GOOGLE_API_KEY" not in os.environ:
#    os.environ["GOOGLE_API_KEY"] = os.getenv("TOKEN")


#model = ChatGoogleGenerativeAI(
#    model="gemini-2.0-flash-001")

lm_studio_url = "http://localhost:1234/v1"

model = ChatOpenAI(
        base_url=lm_studio_url,
        api_key="lm-studio",
        #temperature=0.7,
)

async def main():
    async with MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["services/maths.py"],
                "transport": "stdio",
            },
            "web_search": {
                "command": "python",
                "args": ["services/web_search.py"],
                "transport": "stdio",
            }
        }
    ) as client:
        tools = client.get_tools()
        agent = create_react_agent(model, tools)
        print("Invoking agent...")
        response = await agent.ainvoke({"messages": [("human", "look up what a lemon is")]})
        print("\nAgent Response:")
        pprint(response)

if __name__ == "__main__":
    asyncio.run(main())