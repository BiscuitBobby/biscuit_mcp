from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

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


server_params = StdioServerParameters(
    command="python",
    args=[
        "services/maths.py"
      ]
)

async def run_agent():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)

            # Create and run the agent
            agent = create_react_agent(model, tools)
            agent_response = await agent.ainvoke({"messages": "what is (2+3)x4"})
            return agent_response

# Run the async function
if __name__ == "__main__":
    try:
        result = asyncio.run(run_agent())
        print(result)
    except:
        pass