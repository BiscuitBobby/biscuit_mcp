from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from services.memory import memory
from dotenv import load_dotenv
from pprint import pprint
import asyncio
import os

load_dotenv()

'''
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.getenv("TOKEN")


local_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001")
'''

local_llm = ChatOpenAI(
        base_url="http://localhost:1234/v1",
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
            },
            "memory": {
                "command": "python",
                "args": ["services/memory.py"],
                "transport": "stdio",
            }
        }
    ) as client:
        tools = client.get_tools()
        agent = create_react_agent(local_llm, tools)
        print("Invoking agent...")

        query = input("enter query: ")
        response = await agent.ainvoke({"messages": [("user", query)]})

        print("\nAgent Response:")
        pprint(response)

        response = response['messages'][1].content

        print('\n'+str(response))
        messages = [
                {"role": "user", "content": query},
                {"role": "assistant", "content": response},
        ]
        memory.add(messages, user_id="user", metadata={"category": "chat_history"})

if __name__ == "__main__":
    asyncio.run(main())