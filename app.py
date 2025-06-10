import asyncio
import logging
from src.graphiti.demo import GraphitiDemo

# Configure logging properly for the entire application
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()  # Also output to console
    ]
)

logger = logging.getLogger(__name__)

async def main():
    try:
        logger.info("Starting Graphiti Demo application")
        demo = GraphitiDemo()
        
        logger.info("Running startup sequence")
        await demo.startup()
        
        logger.info("Creating Gradio interface")
        interface = demo.create_gradio_interface()
        
        logger.info("Launching interface")
        interface.launch(debug=True)  # Enable Gradio debug mode
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())