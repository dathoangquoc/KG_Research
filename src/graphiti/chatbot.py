import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json

class GraphitiChatbot:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
    
    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using LangChain-Ollama and available tools"""
        
        # Initialize Ollama chat model
        llm = ChatOllama(
            model="qwen3:1.7b",  # or your preferred model
            temperature=0.7,
        )
        
        # Get available tools
        response = await self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        # Create tools description for the model
        tools_description = "\n".join([
            f"- {tool['name']}: {tool['description']}"
            for tool in available_tools
        ])
        
        # Enhanced system prompt for tool usage
        system_prompt = f"""You are an AI assistant with access to the following tools:

{tools_description}

When you need to use a tool, respond with a JSON object in this exact format:
{{"tool_call": {{"name": "tool_name", "args": {{"param": "value"}}}}}}

If you don't need to use any tools, respond normally with text.

Available tools and their schemas:
{json.dumps(available_tools, indent=2)}
"""

        # Initialize messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]

        # Initial call to Ollama
        response = await llm.ainvoke(messages)
        response_text = response.content

        # Process response and handle tool calls
        final_text = []
        
        # Check if response contains a tool call
        try:
            # Try to parse as JSON for tool calls
            if response_text.strip().startswith('{"tool_call"'):
                tool_call_data = json.loads(response_text)
                tool_name = tool_call_data["tool_call"]["name"]
                tool_args = tool_call_data["tool_call"]["args"]
                
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                
                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                
                # Add tool result to conversation
                messages.append(AIMessage(content=response_text))
                messages.append(HumanMessage(content=f"Tool result: {result.content}"))
                
                # Get follow-up response from Ollama
                follow_up_response = await llm.ainvoke(messages)
                final_text.append(follow_up_response.content)
                
            else:
                # Regular text response, no tool call
                final_text.append(response_text)
                
        except json.JSONDecodeError:
            # If not valid JSON, treat as regular text response
            final_text.append(response_text)
        except KeyError:
            # If JSON doesn't have expected structure, treat as regular text
            final_text.append(response_text)

        return "\n".join(final_text)

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

# Example usage
async def main():
    chatbot = GraphitiChatbot()
    
    try:
        # Connect to your MCP server
        await chatbot.connect_to_server("path/to/your/server.py")
        
        # Process queries
        while True:
            user_input = input("\nEnter your query (or 'quit' to exit): ")
            if user_input.lower() == 'quit':
                break
                
            response = await chatbot.process_query(user_input)
            print(f"\nResponse: {response}")
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        await chatbot.cleanup()

if __name__ == "__main__":
    asyncio.run(main())