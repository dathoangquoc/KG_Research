import os
import docx2txt
import shutil
from datetime import datetime
import asyncio
import traceback

from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

from graphiti_core.nodes import EpisodeType
from .setup_graphiti import setup_graphiti

def upload_docx_file(fileobj):
    """Accept a docx file from the user and return file name and the content"""
    try:
        # Copy file content
        file_name = os.path.basename(fileobj.name) 
        print(f"Processing file: {file_name}")

        os.makedirs("./temps", exist_ok=True)
        path = "./temps/" + file_name

        shutil.copyfile(fileobj.name, path)
        print(f"File copied to: {path}")

        # Get the content of the docx
        loader = Docx2txtLoader(path)
        documents = loader.load_and_split(
            text_splitter=RecursiveCharacterTextSplitter(
                chunk_size=300,
                chunk_overlap=50,
                length_function=len,
            )
        )
        print(f"Loaded {len(documents)} document chunks")
        
        return file_name, documents 
    
    except Exception as e:
        print(f"Error processing file: {e}")
        traceback.print_exc()
        return None, []

async def ingest_docx(fileobj):
    """Add a docx file to the Graphiti Knowledge Graph"""
    if fileobj is None:
        print("No file provided")
        return "No file provided"
    
    graphiti = None
    try:
        print("Setting up Graphiti...")
        graphiti = setup_graphiti()

        # Initialize the graph db with graphiti's indices
        await graphiti.build_indices_and_constraints()
        
        file_name, documents = upload_docx_file(fileobj)
        
        if not documents:
            return "Failed to load document"
            
        print(f"Adding file to graph: {file_name}")
        
        for i, document in enumerate(documents):
            print(f"Adding chunk {i}... Content length: {len(document.page_content)}")
            
            # Add timeout and more detailed logging
            try:
                await graphiti.add_episode(
                        name=file_name,
                        episode_body=document.page_content,
                        source=EpisodeType.text,
                        source_description='Text Passage',
                        group_id="0",
                        reference_time=datetime.now()
                    )
                print(f"Successfully added chunk {i}")

            except Exception as chunk_error:
                print(f"Error adding chunk {i}: {chunk_error}")
                traceback.print_exc()
                return f"Error adding chunk {i}: {str(chunk_error)}"
        
        print(f"Successfully processed all {len(documents)} chunks")
        return f"Successfully ingested {file_name} with {len(documents)} chunks"
        
    except Exception as e:
        print(f"Error in ingest_docx: {e}")
        traceback.print_exc()
        return f"Error: {str(e)}"
        
    finally:
        if graphiti:
            print("Closing Graphiti connection...")
            try:
                await graphiti.close()
                print("Graphiti connection closed")
            except Exception as close_error:
                print(f"Error closing Graphiti: {close_error}")