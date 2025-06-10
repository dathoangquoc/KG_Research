import asyncio
import gradio as gr
from src.graphiti.chatbot import GraphitiChatbot

async def chat_loop(user_query):
    """
    Connects to the Graphiti chatbot server and allows for continuous
    interaction with a user query.
    """
    chatbot = GraphitiChatbot()
    server_script_path = './src/graphiti/mcp_server.py'

    # Step 1: Connect to the server
    try:
        await chatbot.connect_to_server(server_script_path)
    except Exception as e:
        return f"Failed to connect to the server: {e}"

    # Step 2: Process the query
    try:
        response = await chatbot.process_query(user_query)
        return response
    except Exception as e:
        return f"Error processing query: {e}"

# Gradio interface function that wraps the async call
def gradio_chat(user_query):
    response = asyncio.run(chat_loop(user_query))
    return response

# Function to handle file uploads
def process_file(file):
    """
    Processes an uploaded file.
    """
    if file is None:
        return "No file was uploaded."
    
    
    file_path = file.name
    # Add your file processing logic here
    return f"Received file: {file_path}\nFile has been processed successfully."

# Create the chat interface
chat_interface = gr.Interface(
    fn=gradio_chat,
    inputs=gr.Textbox(label="Enter your question:"),
    outputs=gr.Textbox(label="Chatbot Response:"),
    title="Graphiti Chatbot",
    description="Ask questions to the chatbot and get responses based on Graphiti.",
)

# Create the file upload interface
file_interface = gr.Interface(
    fn=process_file,
    inputs=gr.File(label="Upload a file"),
    outputs=gr.Textbox(label="File Processing Result:"),
    title="File Upload",
    description="Upload a file to be processed by the system.",
)

# Create the tabbed interface
tabbed_interface = gr.TabbedInterface(
    [chat_interface, file_interface],
    tab_names=["Chat", "File Upload"]
)

# Launch the Gradio interface
if __name__ == "__main__":
    tabbed_interface.launch()
