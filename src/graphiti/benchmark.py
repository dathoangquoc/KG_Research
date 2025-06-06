"""Benchmarking script for Graphiti."""

import json
import logging
import os
import time

import re
import string

from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
from dotenv import load_dotenv, dotenv_values

# Graphiti imports
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client import LLMConfig
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder import OpenAIRerankerClient
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
from graphiti import setup_graphiti

class GraphitiBenchmark:
    def __init__(self):
        self.graphiti = self.setup_graphiti()
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='./src/graphiti/debug.log'
        )
        self.logger = logging.getLogger(__name__)

    def load_dataset(self, dataset_name: str, subset: tuple = (0, 0)) -> Any:
        """Load a dataset from the datasets folder."""
        with open(f"datasets/{dataset_name}.json", "r") as f:
            dataset = json.load(f)
        
        if subset:
            return dataset[subset[0]:subset[1]]
        else:
            return dataset

    def get_episodes(self, dataset: Any, dataset_name: str) -> List[List[Dict[str, str]]]:
        """Parse the corpus from the dataset into datapoints of episodes."""
        if dataset_name == "2wikimultihopqa":
            datapoints: List[List[Dict[str, str]]] = []
            
            for datapoint in dataset:
                id = datapoint["_id"]
                context = datapoint["context"]
                episodes = []
                
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
                datapoints.append(episodes)

            return datapoints
        else:
            raise NotImplementedError(f"Dataset {dataset_name} not supported.")

    async def create_graph(self, dataset_name: str, subset: tuple = (0, 0)):
        """Create the knowledge graph from the dataset."""
        print("Loading dataset...")
        dataset = self.load_dataset(dataset_name, subset)
        datapoints = self.get_episodes(dataset, dataset_name)
        print(f"Dataset loaded. Data Points: {len(datapoints)}")
        
        graphiti = self.graphiti
        
        try:
            # Initialize the graph db with graphiti's indices
            await graphiti.build_indices_and_constraints()
            
            # Add episodes to the graph
            for index, datapoint in enumerate(datapoints):
                print(f"Adding Data Point {index}/{len(datapoints)}")
                for i, episode in enumerate(datapoint):
                    print(f"Adding Episode {i}/{len(datapoint)} in Data Point {index}")
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
                            f"[added_episode] Title: {episode['title']}, Time: {execution_time:.6f}s, Size: {len(episode["text"])},"
                        )
                        
                    except Exception as e:
                        self.logger.error(f"Error adding episode ({episode['title']}): {e}")

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
        evidence = []
        
        for result_tuple in search_results:
            for result in result_tuple[1]:
                evidence.append(result.name)
        
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
    

    async def benchmark(self, dataset_name: str, subset: tuple = (0, 0)):
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
            tasks = []
            for query in queries:
                tasks.append(self.run_query(query))
            
            for task in tasks:
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

    def normalize_answer(self, answer: str):
        # Lower
        answer = answer.lower()
        
        # Remove punctuation
        exclude = set(string.punctuation)
        answer =  ''.join(ch for ch in answer if ch not in exclude)
        
        # Remove articles (a, an, the)
        answer = re.sub(r'\b(a|an|the)\b', ' ', answer)

        # Remove whitespace
        return ' '.join(answer.split())


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
            ground_truth = [self.normalize_answer(truth) for truth in answer["ground_truth"]]
            predicted_evidence = [self.normalize_answer(evidence) for evidence in answer["evidence"]]
            
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