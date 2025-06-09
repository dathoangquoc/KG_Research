"""Main Gradio application with chatbot and document upload"""

# Python libraries 
from datetime import datetime
import shutil
import os
import traceback

# Graphiti
from graphiti_core.nodes import EpisodeType
from .setup_graphiti import setup_graphiti

# Custom modules
from .document_processor import DocumentProcessor

# Gradio
import gradio as gr


class GraphitiDemo:
    def __init__(self):
        self.graphiti = setup_graphiti()
        self.document_processor = DocumentProcessor()

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
                    # File upload component
                    file_input = gr.File(
                        label="Upload Document",
                        file_types=[".docx", ".txt", ".pdf"],
                        type="filepath"
                    )
                    
                    # Upload button
                    upload_btn = gr.Button("Process Document", variant="primary", size="lg")
                    
                with gr.Column(scale=3):                 
                    # Processing log
                    log_output = gr.Textbox(
                        label="Processing Log",
                        value="",
                        interactive=False,
                        lines=2
                    )
            
            # Event handlers
            upload_btn.click(
                fn=self.process_file_upload,
                inputs=[file_input],
                outputs=[log_output]
            )
            
            # Also trigger on file change for immediate feedback
            file_input.change(
                fn=lambda x: ("File selected: " + os.path.basename(x.name) if x else "No file selected", ""),
                inputs=[file_input],
                outputs=[log_output]
            )
            
            gr.Markdown("---")
            gr.Markdown("**Supported file types:** DOCX, TXT, PDF")
            gr.Markdown("**Note:** Large files may take several minutes to process.")
        
        return interface
            