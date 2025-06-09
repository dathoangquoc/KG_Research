"""Document processing utilities"""

import os
import shutil
from datetime import datetime
import traceback

from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

from graphiti_core.nodes import EpisodeType
from .setup_graphiti import setup_graphiti

CHUNK_SIZE=100
CHUNK_OVERLAP=20

class DocumentProcessor:
    """Handle document processing operations"""

    def __init__(self):
        pass

    def process_file():
        pass

    def _process_docx(self, path):
        """Load a docx file and return chunks"""
        try:
            # Get the content of the docx
            loader = Docx2txtLoader(path)
            documents = loader.load_and_split(
                text_splitter=RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    length_function=len,
                )
            )
            print(f"Loaded {len(documents)} document chunks")
            
            return [document.page_content for document in documents if document.page_content]
        
        except Exception as e:
            print(f"Error processing file: {e}")
            return None

    def _process_txt():
        pass

    
