"""Main Gradio application with chatbot and document upload"""
# Python libraries 
from datetime import datetime
import shutil
import os

# Graphiti
from graphiti_core.nodes import EpisodeType
from setup_graphiti import setup_graphiti

# Custom modules
from document_processor import DocumentProcessor

class GraphitiDemo:
    def __init__(self):
        self.graphiti = setup_graphiti
        self.document_processor = DocumentProcessor

    def process_file_upload(self, fileobj):
        """Process file upload from Gradio interface and add to the knowledge graph"""
        # Process file depending on type
        try:
            # Copy file content
            file_name = os.path.basename(fileobj.name) 
            os.makedirs("./temps", exist_ok=True)
            path = "./temps/" + file_name

            shutil.copyfile(fileobj.name, path)

            # Get the content of the docx
            # Check if file is docx
            chunks = self.document_processor._process_docx(path)
            print(f"Loaded {file_name} with {len(chunks)} chunks")

            # Add to knowledge graph
            self._ingest_episodes(chunks)
        
        except Exception as e:
            print(f"Error processing file: {e}")
            return None, []

            
    async def _ingest_episodes(self, file_name, chunks: list[str]):
        """Add episodes sequentially to the Graphiti Knowledge Graph"""
        try:
            for i, chunk in enumerate(chunks):
                print(f"Adding {i} / {len(chunks)} chunk from {file_name}")
                await self.graphiti.add_episode(
                    name=file_name,
                    episode_body=chunk,
                    source=EpisodeType.text,
                    source_description='Text Passage',
                    group_id="0",
                    reference_time=datetime.now()
                )
        except Exception as e:
            print(e)
        return f"Successfully ingested {file_name}"
        