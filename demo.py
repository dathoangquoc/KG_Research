import gradio as gr
 
from src.graphiti.ingest_docx import ingest_docx

with gr.Blocks() as demo:
    gr.Markdown("## 📄 Upload a Word (.docx) File")
    
    with gr.Row():
        file_input = gr.File(label="Upload .docx file", file_types=[".docx"])
        output = gr.Textbox(label="Extracted Text", lines=20)

    file_input.change(fn=ingest_docx, inputs=file_input, outputs=output)

demo.launch()
