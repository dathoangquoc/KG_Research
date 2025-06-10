import os
import asyncio
import json
from typing import Optional
from contextlib import AsyncExitStack
import logging

from mcp import ClientSession, StdioServerParameters  
from mcp.client.stdio import stdio_client

from dotenv import load_dotenv
import openai

# Configure logging properly
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger(__name__)

load_dotenv("config/.env", override=True)

# LLM configs
llm_api_key = os.environ.get('LLM_API_KEY')
llm_base_url = os.environ.get('LLM_BASE_URL') 
llm_model = os.environ.get('LLM_MODEL')

class GraphitiChatbot:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

        # Configure OpenAI client properly
        self.client = openai.OpenAI(
            api_key=llm_api_key,
            base_url=llm_base_url
        )
        logger.info("Initialized GraphitiChatbot")
        
    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server from a Python or JS script."""
        try:
            logger.info(f"Connecting to server: {server_script_path}")
            
            if server_script_path.endswith('.py'):
                command = 'python'
            elif server_script_path.endswith('.js'):
                command = 'node'
            else:
                raise ValueError("Server script must end with .py or .js")

            server_params = StdioServerParameters(command=command, args=[server_script_path])

            self.stdio, self.write = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )
            await self.session.initialize()

            tools = (await self.session.list_tools()).tools
            logger.info(f"Connected to server with tools: {[tool.name for tool in tools]}")
            
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            raise

    async def process_query(self, query: str):
        """Process a query with OpenAI-compatible API and available tools"""
        try:
            logger.info(f"Processing query: {query}")
            
            messages = [
                {
                    "role": "user", 
                    "content": query
                }
            ]

            # Get available tools if session exists
            available_tools = []
            if self.session:
                try:
                    resources = await self.session.list_tools()
                    available_tools = [{
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema
                        }
                    } for tool in resources.tools]
                    logger.debug(f"Available tools: {[tool['function']['name'] for tool in available_tools]}")
                except Exception as e:
                    logger.warning(f"Could not get tools: {e}")

            # Make initial API call
            response = self.client.chat.completions.create(
                model=llm_model,
                messages=messages,
                tools=available_tools if available_tools else None,
                max_tokens=1000
            )

            logger.debug(f"Got response: {response}")

            # Handle response
            message = response.choices[0].message
            
            # If no tool calls, return the content directly
            if not message.tool_calls:
                content = message.content or "No response generated"
                logger.info(f"Returning direct response: {content[:100]}...")
                return content

            # Handle tool calls
            messages.append(message)
            
            for tool_call in message.tool_calls:
                try:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    logger.info(f"Calling tool {tool_name} with args {tool_args}")
                    
                    # Execute tool call via MCP
                    if self.session:
                        result = await self.session.call_tool(tool_name, tool_args)
                        tool_response = str(result.content) if hasattr(result, 'content') else str(result)
                    else:
                        tool_response = f"Tool {tool_name} not available - no MCP session"
                    
                    # Add tool response to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_response
                    })
                    
                    logger.debug(f"Tool {tool_name} returned: {tool_response[:200]}...")
                    
                except Exception as e:
                    logger.error(f"Error calling tool {tool_name}: {e}")
                    messages.append({
                        "role": "tool", 
                        "tool_call_id": tool_call.id,
                        "content": f"Error: {str(e)}"
                    })

            # Get final response with tool results
            final_response = self.client.chat.completions.create(
                model=llm_model,
                messages=messages,
                max_tokens=1000
            )

            final_content = final_response.choices[0].message.content or "No final response generated"
            logger.info(f"Returning final response: {final_content[:100]}...")
            return final_content

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return f"Error processing query: {str(e)}"