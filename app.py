import asyncio
from src.graphiti.demo import GraphitiDemo

async def main():
    demo = GraphitiDemo()
    await demo.startup()
    interface = demo.create_gradio_interface()
    interface.launch()

if __name__ == "__main__":
    asyncio.run(main())