"""Benchmarking script for Graphiti."""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
from dotenv import load_dotenv, dotenv_values
from tqdm import tqdm

# Graphiti imports
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client import LLMConfig
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder import OpenAIRerankerClient
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

class GraphitiBenchmark:
    def __init__(self):
        dotenv_path = "config/.env"
    
        if os.path.exists(dotenv_path):
            print(f"Found .env file at: {dotenv_path}")
            load_dotenv(dotenv_path, override=True)
        else:
            print(f"No .env file found in '{dotenv_path}'.")

        # Setup logging
        self.setup_logging()
        
        # Neo4j configs
        self.neo4j_uri = os.environ.get('NEO4J_URI')
        self.neo4j_user = os.environ.get('NEO4J_USER')
        self.neo4j_password = os.environ.get('NEO4J_PASSWORD')
        
        # LLM configs
        self.llm_api_key = os.environ.get('LLM_API_KEY')
        self.llm_base_url = os.environ.get('LLM_BASE_URL')
        self.llm_model = os.environ.get('LLM_MODEL')
        
        # Embedder configs
        self.embedder_api_key = os.environ.get('EMBEDDER_API_KEY')
        self.embedder_base_url = os.environ.get("EMBEDDER_BASE_URL")
        self.embedder_model = os.environ.get('EMBEDDER_MODEL')
        self.embedding_dim = os.environ.get('EMBEDDING_DIM')

        self.graphiti = self.setup_graphiti()
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='debug.log'
        )
        self.logger = logging.getLogger(__name__)


    def setup_graphiti(self) -> Graphiti:
        """Initialize Graphiti with configured clients."""
        llm_client = OpenAIClient(
            config=LLMConfig(
                model=self.llm_model,
                api_key=self.llm_api_key,
                base_url=self.llm_base_url,
                small_model=self.llm_model,
                max_tokens=8000
            )
        )
        
        embedder = OpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                embedding_model=self.embedder_model,
                embedding_dim=self.embedding_dim,
                api_key=self.embedder_api_key,
                base_url=self.embedder_base_url
            )
        )
        
        cross_encoder = OpenAIRerankerClient(
            config=LLMConfig(
                model=self.llm_model,
                api_key=self.llm_api_key,
                base_url=self.llm_base_url
            )
        )

        
        return Graphiti(
            uri=self.neo4j_uri,
            user=self.neo4j_user,
            password=self.neo4j_password,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder
        )

    def load_dataset(self, dataset_name: str, subset: int = 0) -> Any:
        """Load a dataset from the datasets folder."""
        with open(f"datasets/{dataset_name}.json", "r") as f:
            dataset = json.load(f)
        
        if subset:
            return dataset[:subset]
        else:
            return dataset

    def get_episodes(self, dataset: Any, dataset_name: str) -> List[Dict[str, str]]:
        """Parse the corpus from the dataset into episodes."""
        if dataset_name == "2wikimultihopqa":
            episodes: List[Dict[str, str]] = []
            
            for datapoint in dataset:
                id = datapoint["_id"]
                context = datapoint["context"]
                
                for passage in context:
                    title = passage[0]
                    text_parts = []
                    for sen in passage[1]:
                        if isinstance(sen, bytes):
                            sen = sen.decode("utf-8")
                        text_parts.append(sen)
                    text = " ".join(text_parts)
                
                if not text:
                    continue

                    
                episodes.append(
                    {
                        "id": id,
                        "title": title,
                        "text": text
                    }
                )
            
            return episodes
        else:
            raise NotImplementedError(f"Dataset {dataset_name} not supported.")

    async def create_graph(self, dataset_name: str, subset: int = 0):
        """Create the knowledge graph from the dataset."""
        print("Loading dataset...")
        dataset = self.load_dataset(dataset_name, subset)
        episodes = self.get_episodes(dataset, dataset_name)
        
        print(f"Dataset loaded. Episodes: {len(episodes)}")
        
        graphiti = self.graphiti
        
        try:
            # Initialize the graph db with graphiti's indices
            await graphiti.build_indices_and_constraints()
            
            # Add episodes to the graph
            for episode in episodes:
                try:
                    start_time = time.perf_counter()
                    await graphiti.add_episode(
                        name=episode["title"],
                        episode_body=episode["text"],
                        source=EpisodeType.text,
                        source_description='Text Passage',
                        group_id=episode["id"],
                        reference_time=datetime.now(timezone.utc)
                    )
                    
                    end_time = time.perf_counter()
                    execution_time = end_time - start_time
                    
                    self.logger.info(
                        f"[added_episode] Title: {episode["title"]}, Time: {execution_time:.6f}s, Size: {len(episode["text"])},"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Error adding episode ({episode["title"]}): {e}")

            print("Graph creation completed!")
            
        finally:
            await graphiti.close()

    def get_questions_and_truths(self, dataset: Any) -> List[Dict]:
        """Get the questions and ground truths from the dataset."""
        questions: List[Dict] = []
        
        for datapoint in dataset:
            id = datapoint["_id"]
            query = datapoint["question"]
            ground_truths = [fact[0].encode("utf-8").decode() for fact in datapoint["supporting_facts"]]
            
            questions.append(
                {
                    "id": id,
                    "question": query,
                    "ground_truths": ground_truths
                }
            )
        return questions
    
    async def run_query(self, question: Dict) -> Dict[str, Any]:
        try:
            graphiti = self.graphiti

            start_time = time.perf_counter()
            # Search for relevant information
            print(f"Searching for {question["question"]} in {question["id"]}")

            search_results = await graphiti.search_(
                query=question["question"],
                group_ids=[question["id"]],
                config=NODE_HYBRID_SEARCH_RRF
            )
            
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            # Extract evidence from search results
            for result in search_results:
                print("Search Result:", result)
            evidence = [result["name"] for result in search_results]
            
            self.logger.info(
                f"[query] Question: '{question["question"]}', Time: {execution_time:.6f}s, Evidence count: {len(question["ground_truths"])}"
            )
            
            return {
                "question": question["question"],
                "answer": "",
                "evidence": evidence,
                "ground_truth": question["ground_truths"],
                "execution_time": execution_time
            }
            
        except Exception as e:
            self.logger.error(f"Error processing question[question] '{question["question"]}': {e}")
            self.logger.error(f"Error processing question '{question}': {e}")
            return {
                "question": question["question"],
                "answer": "",
                "evidence": [],
                "ground_truth": question["ground_truths"],
                "execution_time": 0.0
            }
    

    async def benchmark(self, dataset_name: str, subset: int = 0):
        """Benchmark the knowledge graph on queries."""
        print("Loading dataset...")
        dataset = self.load_dataset(dataset_name, subset)
        queries = self.get_questions_and_truths(dataset)
        
        print(f"Dataset loaded. Queries: {len(queries)}")
        
        graphiti = self.setup_graphiti()
        results = []
        
        try:
            
            # Process all queries
            print("Processing queries...")
            for query in queries:
                tasks = []
                print(query)
                tasks.append(self.run_query(query))
            
            for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Querying"):
                result = await task
                results.append(result)
            
            # Save results
            results_path = f"benchmark_results/graphiti_{dataset_name}.json"
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)
            
            print(f"Benchmark completed! Results saved to {results_path}")
            
        finally:
            await graphiti.close()

    def compute_scores(self, dataset_name: str):
        """Compute and display benchmark scores."""
        results_path = f"benchmark_results/graphiti_{dataset_name}.json"
        
        try:
            with open(results_path, "r") as f:
                answers = json.load(f)
        except FileNotFoundError:
            print(f"Results file not found: {results_path}")
            return
        
        # Compute retrieval metrics
        retrieval_scores: List[float] = []
        execution_times: List[float] = []
        
        for answer in answers:
            ground_truth = answer["ground_truth"]
            predicted_evidence = answer["evidence"]
            
            if ground_truth:  # Avoid division by zero
                p_retrieved = len(set(ground_truth).intersection(set(predicted_evidence))) / len(set(ground_truth))
            else:
                p_retrieved = 0.0
            
            retrieval_scores.append(p_retrieved)
            execution_times.append(answer.get("execution_time", 0.0))
        
        # Print results
        print(f"\n=== Benchmark Results for {dataset_name} ===")
        print(f"Total queries: {len(answers)}")
        print(f"Average retrieval score: {np.mean(retrieval_scores):.4f}")
        print(f"Percentage of queries with perfect retrieval: {np.mean([1 if s == 1.0 else 0 for s in retrieval_scores]):.4f}")
        print(f"Average execution time: {np.mean(execution_times):.4f}s")
        print(f"Total execution time: {np.sum(execution_times):.4f}s")

async def main():
    benchmark = GraphitiBenchmark()
    
    await benchmark.create_graph(dataset_name="2wikimultihopqa", subset=5)
    
    await benchmark.benchmark(dataset_name="2wikimultihopqa", subset=5)
    
    benchmark.compute_scores(dataset_name="2wikimultihopqa")


if __name__ == "__main__":
    asyncio.run(main())