import asyncio

import gradio as gr
from src.graphiti.demo import GraphitiDemo 

demo = GraphitiDemo()

# Function to handle file uploads
def process_file(fileobj):
    # Create a new event loop instead of trying to get the current one
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Run the async function and get the result
    result = loop.run_until_complete(demo.process_file_upload(fileobj))
    loop.close()
    return result

# Similar fix for the chat function
def gradio_chat(user_query):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(demo.ask(user_query))
    loop.close()
    return result

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
