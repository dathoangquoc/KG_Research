"""MCP Server for Graphiti"""

# MCP imports
from mcp.server.fastmcp import FastMCP

# Fix path for direct script execution
import sys
import os
# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Now use absolute import
from src.graphiti.setup_graphiti import setup_graphiti

mcp = FastMCP()

class GraphitiServer():
    def __init__(self):
        self.graphiti = setup_graphiti()
        
    @mcp.tool()
    async def search_facts(self, query: str) -> str:
        """Search for facts from the Knowledge Graph
        
        Args:
            query: the query to search for
        """
        
        results = await self.graphiti.search(
            query=query,
            group_ids=["0"],
            num_results=5
        )

        # Format facts into readable string
        facts = [result.fact for result in results]
        return "\n".join(facts)

# Add a server instance and entry point
server = GraphitiServer()

# Run the MCP server when executed directly
if __name__ == "__main__":
    mcp.run(transport='stdio')