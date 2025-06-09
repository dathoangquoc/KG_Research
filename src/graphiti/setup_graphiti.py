import os
from dotenv import load_dotenv

from graphiti_core import Graphiti
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client import LLMConfig
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder import OpenAIRerankerClient

def setup_graphiti(dotenv_path: str = "config/.env") -> Graphiti:
    """Initialize Graphiti with configured clients."""
    if os.path.exists(dotenv_path):
            print(f"Found .env file at: {dotenv_path}")
            load_dotenv(dotenv_path, override=True)
    else:
        print(f"No .env file found in '{dotenv_path}'.")

    # Neo4j configs
    neo4j_uri = os.environ.get('NEO4J_URI')
    neo4j_user = os.environ.get('NEO4J_USER')
    neo4j_password = os.environ.get('NEO4J_PASSWORD')
    
    # LLM configs
    llm_api_key = os.environ.get('LLM_API_KEY')
    llm_base_url = os.environ.get('LLM_BASE_URL')
    llm_model = os.environ.get('LLM_MODEL')
    
    # Embedder configs
    embedder_api_key = os.environ.get('EMBEDDER_API_KEY')
    embedder_base_url = os.environ.get("EMBEDDER_BASE_URL")
    embedder_model = os.environ.get('EMBEDDER_MODEL')
    embedding_dim = os.environ.get('EMBEDDING_DIM')

    llm_client = OpenAIClient(
        config=LLMConfig(
            model=llm_model,
            api_key=llm_api_key,
            base_url=llm_base_url,
            small_model=llm_model,
            max_tokens=8000
        )
    )
    
    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            embedding_model=embedder_model,
            embedding_dim=embedding_dim,
            api_key=embedder_api_key,
            base_url=embedder_base_url
        )
    )
    
    cross_encoder = OpenAIRerankerClient(
        config=LLMConfig(
            model=llm_model,
            api_key=llm_api_key,
            base_url=llm_base_url
        )
    )

    
    return Graphiti(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder
    )