"""MCP Server for Graphiti"""

# MCP imports
from mcp.server.fastmcp import FastMCP

import logging

# Fix path for direct script execution
import sys
import os
# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Now use absolute import
from src.graphiti.setup_graphiti import setup_graphiti

logger = logging.getLogger(__name__)

class GraphitiServer():
    def __init__(self):
        self.graphiti = setup_graphiti()
        self.mcp = FastMCP()
        self._register_tools()
        
    def _register_tools(self):
        """Register tools with the MCP server"""
        
        @self.mcp.tool()
        async def search_facts(query: str) -> str:
            """Search for facts from the Knowledge Graph
            
            Args:
                query: the query to search for
            """
            try:
                logger.info(f"Searching for facts with query: {query}")
                
                # Ensure graphiti is initialized
                graph = self.graphiti

                results = await graph.search(
                    query=query,
                    group_ids=["0"],
                    num_results=3
                )

                # Format facts into readable string
                if not results:
                    logger.info("No results found")
                    return "No facts found for the given query."
                    
                facts = [result.fact for result in results]
                formatted_facts = "\n".join([f"- {fact}" for fact in facts])
                
                logger.info(f"Found {len(facts)} facts")
                return formatted_facts
                
            except Exception as e:
                logger.error(f"Error searching facts: {e}", exc_info=True)
                return f"Error searching facts: {str(e)}"
            
    def run(self):
        """Run the MCP server"""
        logger.info("Starting MCP Server...")
        try:
            self.mcp.run(transport='stdio')
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}", exc_info=True)
            raise

# Create and run server when executed directly
if __name__ == "__main__":
    server = GraphitiServer()
    server.run()