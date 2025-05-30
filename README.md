# Research and Test Recent Advancements in Knowledge Graph - LLMs Synergy

This repo is part of an ongoing effort to test Knowledge Graph implementations for enhanced LLMs reasoning capability.
Included approaches:
- [Graphiti](https://github.com/getzep/graphiti)
- [Fast-GraphRAG](https://github.com/circlemind-ai/fast-graphrag) (wip)

## Folder Structure

```
.
├── config/             Environment configs
├── dataset/            Datasets for benchmarking
├── docs/               Documentations
├── pics/               Diagrams
├── scripts/            Scripts for running benchmarks
├── src/                Source code for frameworks
│   ├── graphiti/       
│   └── fastgraph/
├── requirements/       Dependencies list for frameworks
└── README.md
```

## How to run the benchmarks

Dependencies:
- A script in provided for quick virtual environment setup in `scripts/init_graphiti.sh`
- You can create the environment and installing the dependencies yourself using `requirements/graphiti_requirements.txt`

Neo4j Database:
- You can connect to instance of AuraDB or host a local one using Docker with `scripts/start_neo4j.sh`

Run the benchmark `python3 main.py`

## Environment Variables

Place your .env in /config
Example .env for running with an LLM and embedding model hosted locally through Ollama

```
# neo4j configs
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

# LLM configs
LLM_API_KEY = "dummy"
LLM_BASE_URL = "http://localhost:11434/v1"
LLM_MODEL = "qwen3:8b"

# Embedder configs
EMBEDDER_API_KEY = "dummy"
EMBEDDER_BASE_URL = "http://localhost:11434/v1"
EMBEDDER_MODEL = "snowflake-arctic-embed2:latest"
EMBEDDING_DIM = 1024
```
