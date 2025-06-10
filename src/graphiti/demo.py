"""Main Gradio application with chatbot and document upload"""

# Python libraries 
from datetime import datetime
import shutil
import os
import traceback
import logging

# Graphiti
from graphiti_core.nodes import EpisodeType
from .setup_graphiti import setup_graphiti

# Ollama
import ollama

# Custom modules
from .document_processor import DocumentProcessor
from .mcp_server import GraphitiServer
from .chatbot import GraphitiChatbot

# Gradio
import gradio as gr

logger = logging.getLogger(__name__)

class GraphitiDemo:
    def __init__(self):
        self.graphiti = setup_graphiti()
        self.document_processor = DocumentProcessor()
        self.graphiti_server = GraphitiServer()
        self.chatbot = GraphitiChatbot()

    async def startup(self):
        """Run this before launching the Gradio app to initialize connections"""
        await self.chatbot.connect_to_server("./src/graphiti/mcp_server.py")

    async def chatbot_respond(self, user_msg, chat_history):
        """Handle chatbot interaction from UI"""
        try:
            if user_msg.lower() == 'quit':
                return chat_history + [[user_msg, "Goodbye!"]]
            response = await self.chatbot.process_query(user_msg)
            return chat_history + [[user_msg, response]]
        except Exception as e:
            print("Error in chatbot response:", e)
            return chat_history + [[user_msg, f"Error: {str(e)}"]]

    async def process_file_upload(self, fileobj):
        """Process file upload from Gradio interface and add to the knowledge graph"""
        # TODO: Process file depending on type
        if fileobj is None:
            return "No file uploaded"

        try:
            # Copy file content
            file_name = os.path.basename(fileobj.name) 
            os.makedirs("./temps", exist_ok=True)
            path = "./temps/" + file_name

            shutil.copyfile(fileobj.name, path)

            # Get the content of the docx
            # TODO: Check if file is docx
            chunks = self.document_processor._process_docx(path)

            if chunks is None:
                return "Error processing file. Check console bruh"
            print(f"Queued {file_name} with {len(chunks)} chunks")

            log = await self._ingest_episodes(file_name, chunks)
            return log
                    
        except Exception as e:
            print("Error processing upload: ", e)
            return e

    async def _ingest_episodes(self, file_name, chunks: list[str]):
        """Add episodes sequentially to the Graphiti Knowledge Graph"""
        try:
            for i, chunk in enumerate(chunks):
                print(f"Adding {i} / {len(chunks)} chunks from {file_name}")
                await self.graphiti.add_episode(
                    name=file_name,
                    episode_body=chunk,
                    source=EpisodeType.text,
                    source_description='Text Passage',
                    group_id="0",
                    reference_time=datetime.now()
                )
        except Exception as e:
            print("Error ingesting episodes: ", e)
            traceback.print_exc()
            return f"Error ingesting episodes: {e}"
        return f"Successfully ingested {file_name}"
    
    def create_gradio_interface(self):
        with gr.Blocks(title="Graphiti Knowledge Graph Demo", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🧠 Graphiti Knowledge Graph Demo")
            gr.Markdown("Upload documents to add them to the knowledge graph for processing and analysis.")
            
            with gr.Row():
                with gr.Column(scale=2):
                    file_input = gr.File(label="Upload Document", file_types=[".docx", ".txt", ".pdf"], type="filepath")
                    upload_btn = gr.Button("Process Document", variant="primary", size="lg")
                with gr.Column(scale=3):
                    log_output = gr.Textbox(label="Processing Log", value="", interactive=False, lines=2)

            upload_btn.click(
                fn=self.process_file_upload,
                inputs=[file_input],
                outputs=[log_output]
            )
            file_input.change(
                fn=lambda x: ("File selected: " + os.path.basename(x.name) if x else "No file selected", ""),
                inputs=[file_input],
                outputs=[log_output]
            )

            gr.Markdown("---")
            gr.Markdown("### 💬 Ask the Chatbot (integrated with Ollama + Graphiti)")
            
            with gr.Column():
                chatbot_ui = gr.Chatbot(height=400, type='messages')
                user_input = gr.Textbox(
                    label="Ask something...", 
                    placeholder="Type your message here...",
                    lines=1
                )
                
                # Handle user input submission
                user_input.submit(
                    fn=self.chatbot_respond,
                    inputs=[user_input, chatbot_ui],
                    outputs=[chatbot_ui],
                    queue=True
                ).then(
                    fn=lambda: "",  # Clear input after submission
                    inputs=[],
                    outputs=[user_input]
                )

            gr.Markdown("**Note:** Chatbot uses local Ollama + optionally queries Graphiti for factual search.")

        return interface