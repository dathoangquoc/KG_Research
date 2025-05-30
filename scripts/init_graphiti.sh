#!/bin/bash 

# Create and activate a new python virtual environment
python3 -m venv /src/graphiti/.venv
source /src/graphiti/.venv/bin/activate

# Install packages
pip install -r /requirements/graphiti_requirements.txt

# Start a local neo4j database with Docker
bash /scripts/start_neo4j.sh