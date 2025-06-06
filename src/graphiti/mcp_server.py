"""MCP Server for Graphiti"""

import asyncio
import json
import logging
import os
import time
import re
import string

from dotenv import load_dotenv
from datetime import datetime, timezone
from typing import Any, Dict, List

# MCP imports
from mcp.server.fastmcp import FastMCP

# Graphiti imports
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client import LLMConfig
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder import OpenAIRerankerClient
from graphiti import setup_graphiti

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
        