# Main application with chatbot and document upload

# Python libraries 
from datetime import datetime
import shutil
import os
import traceback
import logging
import asyncio

# Graphiti
from graphiti_core.nodes import EpisodeType
from .setup_graphiti import setup_graphiti

# Custom modules
from .document_processor import DocumentProcessor
from .mcp_server import GraphitiServer
from .chatbot import GraphitiChatbot

logger = logging.getLogger(__name__).setLevel(logging.CRITICAL)
server_script_path = './src/graphiti/mcp_server.py'

class GraphitiDemo:
    def __init__(self):
        self.graphiti = setup_graphiti()
        self.document_processor = DocumentProcessor()
        self.graphiti_server = GraphitiServer()
        self.chatbot = GraphitiChatbot()

    async def process_file_upload(self, fileobj):
        """Process file upload from Gradio interface and add to the knowledge graph"""
        # TODO: Process file depending on type
        if fileobj is None:
            return "No file uploaded"

        try:
            # Initialize the graph database with graphiti's indices. This only needs to be done once.
            await self.graphiti.build_indices_and_constraints()

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

    async def ask(self, query):
        await self.chatbot.connect_to_server(server_script_path)
        response = await self.chatbot.process_query(query)
        return response