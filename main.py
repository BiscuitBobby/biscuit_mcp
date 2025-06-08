from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Annotated, List, Tuple, Union, Literal
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from dotenv import load_dotenv
import operator
import asyncio
import re
import os

load_dotenv()

local_llm = ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        #temperature=0.7,
)

# Define researcher agent tool
@tool
async def researcher(query: str) -> str:
    """
    A researcher agent that conducts thorough research on given topics using web search.

    Args:
        query: The research question or topic to investigate
    """
    try:
        print("Researcher called")
        researcher_llm = local_llm

        async with MultiServerMCPClient(
            {
                "web_search": {
                    "command": "python",
                    "args": ["services/web_search.py"],
                    "transport": "stdio",
                }
            }
        ) as client:
            web_tools = client.get_tools()

            researcher_agent = create_react_agent(researcher_llm, web_tools)

            research_prompt = f"""
            You are a professional researcher with access to web search tools.
            Conduct thorough research on the following topic using available tools:

            Topic: {query}

            Make it brief.
            """

            result = await researcher_agent.ainvoke({"messages": [("user", research_prompt)]})

            # Extract the final response
            if 'messages' in result and result['messages']:
                final_message = result['messages'][-1]
                if hasattr(final_message, 'content'):
                    return f"Research Report (with Web Search):\n{final_message.content}"
                else:
                    return f"Research Report (with Web Search):\n{str(final_message)}"
            else:
                return f"Research Report (with Web Search):\n{str(result)}"

    except Exception as e:
        return f"Research error: {str(e)}"

# Define advisor agent tool
@tool
async def advisor(context: str) -> str:
    """
    An advisor agent that provides strategic advice and recommendations.

    Args:
        context: The situation or problem requiring advice
    """

    advisor_llm = local_llm

    advisory_prompt = f"""
    You are a strategic advisor with expertise across multiple domains.
    Analyze the following situation and provide advice:

    Situation: {context}

    Be practical and Brief.
    """

    try:
        print("Advisor called")
        response = await advisor_llm.ainvoke(advisory_prompt)
        return f"Advisory Report:\n{response.content}"
    except Exception as e:
        return f"Advisory error: {str(e)}"


# Create tools list
agents = [researcher, advisor]

# Create the main agent
exec_agent = create_react_agent(local_llm, agents)

class PlanExecute(TypedDict):
    """
    Represents the state of the execution workflow.

    Attributes:
        input (str): The user's input or objective.
        plan (List[str]): A list of steps to achieve the objective.
        past_steps (List[Tuple]): A list of tuples representing completed steps and their results.
        response (str): The final response to the user.
    """
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str

# Planning Step
class Plan(BaseModel):
    """
    Represents a plan consisting of steps to achieve an objective.

    Attributes:
        steps (List[str]): A list of steps to follow, sorted in the required order.
    """
    steps: List[str] = Field(description="different steps to follow, should be in sorted order")


planner = local_llm.with_structured_output(Plan)



# Re-Plan Step
class Response(BaseModel):
    """
    Represents a response to the user.

    Attributes:
        response (str): The final response message.
    """
    response: str

class Act(BaseModel):
    """
    Represents an action to perform, which can be either a response or a new plan.

    Attributes:
        action (Union[Response, Plan]): The action to perform. Use `Response` to respond to the user or `Plan` to continue executing steps.
    """
    action: Union[Response, Plan] = Field(description="Action to perform. If you want to respond to user, use Response. If you need to further use tools to get the answer, use Plan.")

replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)

replanner = replanner_prompt | local_llm.with_structured_output(Act)

# Create the Graph
async def agent_node(state: PlanExecute) -> PlanExecute:
    """
    Executes the first step in the plan using the agent.

    Args:
        state (PlanExecute): The current state of the workflow.

    Returns:
        PlanExecute: The updated state with the result of the executed step.
    """
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"For the following plan:\n{plan_str}\n\nYou are tasked with executing step {1}, {task}."
    agent_response = await exec_agent.ainvoke({"messages": [("user", task_formatted)]})
    return {"past_steps": [(task, agent_response["messages"][-1].content)]}

async def create_plan(objective: str) -> List[str]:
    """
    Creates a plan by prompting the LLM and parsing the response.
    """
    planning_prompt = f"""For the given objective, come up with a simple step by step plan.
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps.
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Objective: {objective}

Please provide your response as a numbered list of steps, one step per line, starting each step with a number followed by a period and a space.

Example format:
1. Research the topic X
2. Analyze the data from step 1
3. Provide final recommendation

Your plan:"""

    response = await local_llm.ainvoke(planning_prompt)

    # Parse the response to extract steps
    content = response.content if hasattr(response, 'content') else str(response)

    # Extract numbered steps using regex
    steps = []
    lines = content.strip().split('\n')

    for line in lines:
        line = line.strip()
        # Match patterns like "1. Step description" or "1) Step description"
        match = re.match(r'^\d+[\.\)]\s*(.+)', line)
        if match:
            steps.append(match.group(1).strip())

    # If no numbered steps found, try to split by lines and clean up
    if not steps:
        for line in lines:
            line = line.strip()
            if line and not line.startswith('Example') and not line.startswith('Your plan'):
                # Remove leading numbers, dashes, or bullets
                cleaned_line = re.sub(r'^[\d\-\*\•]\s*[\.\)]*\s*', '', line)
                if cleaned_line:
                    steps.append(cleaned_line)

    return steps if steps else ["Complete the given objective"]

async def plan_node(state: PlanExecute) -> PlanExecute:
    """
    Generates a plan based on the user's input.

    Args:
        state (PlanExecute): The current state of the workflow.

    Returns:
        PlanExecute: The updated state with the generated plan.
    """

    steps = await create_plan(state["input"])

    print(f"Generated plan steps: {steps}")

    return {"plan": steps}

# Also fix the replanner to use the same approach
async def create_replan(input_text: str, original_plan: List[str], past_steps: List[Tuple]) -> Union[str, List[str]]:
    """
    Creates a replan or final response.
    """
    past_steps_text = "\n".join([f"- {step}: {result}" for step, result in past_steps])
    original_plan_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(original_plan)])

    replan_prompt = f"""For the given objective, come up with a simple step by step plan.
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps.
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input_text}

Your original plan was this:
{original_plan_text}

You have currently done the follow steps:
{past_steps_text}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with "FINAL_RESPONSE:" followed by your final answer. Otherwise, provide a numbered list of remaining steps that still NEED to be done. Do not return previously done steps as part of the plan.

Your response:"""

    response = await local_llm.ainvoke(replan_prompt)
    content = response.content if hasattr(response, 'content') else str(response)

    # Check if this is a final response
    if "FINAL_RESPONSE:" in content:
        final_response = content.split("FINAL_RESPONSE:", 1)[1].strip()
        return final_response  # Return string for final response

    # Otherwise, parse as new plan steps
    steps = []
    lines = content.strip().split('\n')

    for line in lines:
        line = line.strip()
        match = re.match(r'^\d+[\.\)]\s*(.+)', line)
        if match:
            steps.append(match.group(1).strip())

    if not steps:
        for line in lines:
            line = line.strip()
            if line and not line.lower().startswith('your response'):
                cleaned_line = re.sub(r'^[\d\-\*\•]\s*[\.\)]*\s*', '', line)
                if cleaned_line:
                    steps.append(cleaned_line)

    return steps if steps else [content.strip()]

async def replan_node(state: PlanExecute) -> PlanExecute:
    """
    Updates the plan based on the results of previous steps.

    Args:
        state (PlanExecute): The current state of the workflow.

    Returns:
        PlanExecute: The updated state with the new plan or final response.
    """
    result = await create_replan(state["input"], state["plan"], state["past_steps"])

    if isinstance(result, str):
        # Final response
        return {"response": result}
    else:
        # New plan steps
        return {"plan": result}

def should_end(state: PlanExecute) -> Literal["agent_node", END]:
    """
    Determines whether the workflow should end or continue.

    Args:
        state (PlanExecute): The current state of the workflow.

    Returns:
        Literal["agent_node", END]: Returns `END` if the workflow should terminate, otherwise returns `"agent_node"`.
    """
    if "response" in state and state["response"]:
        return END
    else:
        return "agent_node"

workflow = StateGraph(PlanExecute)
workflow.add_node("plan_node", plan_node)
workflow.add_node("agent_node", agent_node)
workflow.add_node("replan_node", replan_node)

workflow.add_edge(START, "plan_node")
workflow.add_edge("plan_node", "agent_node")
workflow.add_edge("agent_node", "replan_node")
workflow.add_conditional_edges("replan_node", should_end, ["agent_node", END])

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()

from IPython.display import Image, display
display(Image(app.get_graph(xray=True).draw_mermaid_png()))

async def main():
    config = {"recursion_limit": 30}
    inputs = {"input": input("prompt: ")}
    async for event in app.astream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(v)

if __name__ == "__main__":
    asyncio.run(main())
