# math_server.py
from mcp.server.fastmcp import FastMCP
from audit import log
mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    log(f"add({a},{b})")
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    log(f"multiply({a},{b})")
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")