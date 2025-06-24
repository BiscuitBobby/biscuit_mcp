import operator
from typing import Annotated, List, Literal, TypedDict
from pprint import pprint

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.combine_documents.reduce import (
    acollapse_docs,
    split_list_of_docs,
)
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph


class OverallState(TypedDict):
    contents: List[str]
    summaries: Annotated[list, operator.add]
    collapsed_summaries: List[Document]
    final_summary: str


class SummaryState(TypedDict):
    content: str


async def create_document_summarizer(llm, documents: List[Document], chunk_size: int = 1000, token_max: int = 1000):
    """
    Create and execute a document summarization pipeline using LangGraph.
    
    Args:
        llm: The language model to use for summarization
        documents: List of Document objects to summarize
        chunk_size: Size of text chunks for splitting (default: 1000)
        token_max: Maximum tokens for summary collapse threshold (default: 1000)
    
    Returns:
        str: Final comprehensive summary
    """
    
    # Define prompts
    map_prompt = ChatPromptTemplate.from_messages(
        [("system", "Write a concise summary of the following:\\n\\n{context}")]
    )
    
    reduce_template = """
    The following are summaries:
    {docs}
    Synthesize these summaries into a comprehensive overview.
    """
    reduce_prompt = ChatPromptTemplate([("human", reduce_template)])
    
    # Split documents
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=0
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Generated {len(split_docs)} documents.")
    
    # Define token length function
    def length_function(docs: List[Document]) -> int:
        return sum(llm.get_num_tokens(doc.page_content) for doc in docs)
    
    # Node functions
    async def generate_summary(state: SummaryState):
        """Generates a summary for a single chunk of text"""
        prompt = map_prompt.invoke({"context": state["content"]})
        response = await llm.ainvoke(prompt)
        return {"summaries": [response.content]}
    
    def map_summaries(state: OverallState):
        """Maps out over the documents to generate summaries"""
        return [
            Send("generate_summary", {"content": content}) 
            for content in state["contents"]
        ]
    
    def collect_summaries(state: OverallState):
        """Collects the summaries from all chunks"""
        return {
            "collapsed_summaries": [Document(page_content=summary) 
                                   for summary in state["summaries"]]
        }
    
    async def _reduce(input_docs) -> str:
        """Helper function to reduce summaries"""
        docs_content = "\n".join([doc.page_content for doc in input_docs])
        prompt = reduce_prompt.invoke({"docs": docs_content})
        response = await llm.ainvoke(prompt)
        return response.content
    
    async def collapse_summaries(state: OverallState):
        """Combines the summaries if they exceed the maximum token limit"""
        doc_lists = split_list_of_docs(
            state["collapsed_summaries"], length_function, token_max
        )
        results = []
        for doc_list in doc_lists:
            collapsed_content = await _reduce(doc_list)
            results.append(Document(page_content=collapsed_content))
        
        return {"collapsed_summaries": results}
    
    async def generate_final_summary(state: OverallState):
        """Generates the final comprehensive summary"""
        response = await _reduce(state["collapsed_summaries"])
        return {"final_summary": response}
    
    def should_collapse(state: OverallState) -> Literal["collapse_summaries", "generate_final_summary"]:
        """Determines if summaries should be collapsed or final summary generated"""
        num_tokens = length_function(state["collapsed_summaries"])
        if num_tokens > token_max:
            return "collapse_summaries"
        else:
            return "generate_final_summary"
    
    # Build the graph
    graph = StateGraph(OverallState)
    graph.add_node("generate_summary", generate_summary)
    graph.add_node("collect_summaries", collect_summaries)
    graph.add_node("collapse_summaries", collapse_summaries)
    graph.add_node("generate_final_summary", generate_final_summary)
    
    # Add edges
    graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
    graph.add_edge("generate_summary", "collect_summaries")
    graph.add_conditional_edges("collect_summaries", should_collapse)
    graph.add_conditional_edges("collapse_summaries", should_collapse)
    graph.add_edge("generate_final_summary", END)
    
    # Compile and run the graph
    app = graph.compile()
    
    final_step = None
    async for step in app.astream(
        {"contents": [doc.page_content for doc in split_docs]},
        {"recursion_limit": 20},
    ):
        print(f"Processing step: {list(step.keys())}")

        if "generate_summary" in step:
            summaries = step["generate_summary"].get("summaries", [])
            for i, summary in enumerate(summaries):
                preview = summary[:200].replace("\n", " ") + ("..." if len(summary) > 200 else "")
                print(f"  Chunk {i} summary preview: {preview}")

        final_step = step
    
    return final_step["generate_final_summary"]["final_summary"]


def simple_summarize(llm, documents: List[Document]):
    """
    Simple summarization function for smaller documents that don't need map-reduce.
    
    Args:
        llm: The language model to use for summarization
        documents: List of Document objects to summarize
    
    Returns:
        str: Summary of the documents
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", "Write a concise summary of the following:\\n\\n{context}")]
    )
    
    chain = create_stuff_documents_chain(llm, prompt)
    result = chain.invoke({"context": documents})
    
    return result.strip()


# Example usage function
async def summarize_documents(llm, documents: List[Document], chunk_size: int = 1000, use_map_reduce: bool = True):
    """
    Main function to summarize documents with configurable parameters.
    
    Args:
        llm: The language model to use
        documents: List of documents to summarize
        chunk_size: Size of chunks for map-reduce (default: 1000)
        use_map_reduce: Whether to use map-reduce or simple summarization (default: True)
    
    Returns:
        str: Final summary
    """
    if use_map_reduce:
        summary = await create_document_summarizer(llm, documents, chunk_size)
    else:
        summary = simple_summarize(llm, documents)
    
    return summary

#import os
import asyncio
#from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader


local_llm = ChatOpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        #temperature=0.7,
)


'''
load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.getenv("TOKEN")


local_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001")
'''


async def get_summary(doc_path, model=local_llm):

        loader = PyPDFLoader(doc_path)
        pages = []
        async for page in loader.alazy_load():
                pages.append(page)

        # For map-reduce summarization:
        final_summary = await summarize_documents(
        llm=model, 
        documents=pages, 
        chunk_size=800,
        use_map_reduce=True
        )

        # For simple summarization:
        # simple_summary = await summarize_documents(
        # llm=local_llm, 
        # documents=docs, 
        # use_map_reduce=False
        # )

        print("Final Summary:")
        pprint(final_summary)

if __name__ == "__main__":
    asyncio.run(get_summary("/home/biscuitbobby/Documents/mcp/services/docs/Independent_Sugar_Corporation_Limited_vs_Girish_Sriram_Juneja_on_29_January_2025.PDF", local_llm))
