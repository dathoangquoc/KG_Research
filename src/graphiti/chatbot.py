import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import sys
from ollama import chat as ollama_chat

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, server_script_path: str):
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
        messages = [{"role": "user", "content": query}]
        final_response = []

        # Get tools from the MCP server
        tool_response = await self.session.list_tools()
        tools = tool_response.tools

        # Convert MCP tools to Ollama-compatible tool definitions
        ollama_tools = []
        for tool in tools:
            ollama_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            })

        # First call to Ollama
        response = ollama_chat(
            model="qwen3:1.7b",  # Or your preferred model
            messages=messages,
            tools=ollama_tools,
        )

        content = response['message'].get('content')
        if content:
            final_response.append(content)

        # Handle tool calls (if any)
        tool_calls = response['message'].get('tool_calls', [])
        for call in tool_calls:
            tool_name = call['function']['name']
            tool_args = json.loads(call['function']['arguments'])

            # Execute tool
            result = await self.session.call_tool(tool_name, tool_args)

            # Append tool result to messages and re-query Ollama
            messages.append({
                "role": "assistant",
                "tool_call_id": call['id'],
                "content": None,
                "tool_calls": [call]
            })

            messages.append({
                "role": "tool",
                "tool_call_id": call['id'],
                "name": tool_name,
                "content": result.content
            })

            # Follow-up query with tool result
            response = ollama_chat(
                model="llama3.1",
                messages=messages,
            )

            final_response.append(response['message'].get('content', ''))

        return "\n".join(final_response)

    async def chat_loop(self):
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == 'quit':
                    break
                response = await self.process_query(query)
                print("\n" + response)
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
