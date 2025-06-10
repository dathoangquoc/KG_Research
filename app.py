import asyncio
import gradio as gr
from src.graphiti.demo import GraphitiDemo 

demo = GraphitiDemo()

# Gradio interface function that wraps the async call
def gradio_chat(user_query):
    response = asyncio.run(demo.ask(user_query))
    return response

# Function to handle file uploads
def process_file(fileobj):
    response = asyncio.run(demo.process_file_upload(fileobj))
    return response

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
