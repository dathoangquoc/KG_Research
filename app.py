from src.graphiti.demo import GraphitiDemo

if __name__ == "__main__":
    demo = GraphitiDemo()
    interface = demo.create_gradio_interface()
    interface.launch()