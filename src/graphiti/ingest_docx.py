import os
import docx2txt
import shutil

def upload_file(fileobj):
    """Accept a docx file from the user and return file name and the content"""
    try:
        # Copy file content
        file_name = os.path.basename(fileobj)
        path = "/home/ubuntu/temps/" + file_name  
        shutil.copyfile(fileobj.name, path)

        # Get the content of the docx
        content = docx2txt.process(path)
        content = content.split()
        
        yield file_name, content
    
    except Exception as e:
        print(f"Error processing {path}: {e}")

def ingest_docx(path: str):
    """Add a docx file to the Graphiti Knowledge Graph"""