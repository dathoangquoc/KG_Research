"""Benchmarking script for Graphiti on 2WikiMultiHopQA dataset."""

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm import tqdm

# Graphiti imports
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client import LLMConfig
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder import OpenAIRerankerClient

class GraphitiBenchmark:
    def __init__(self):
        load_dotenv(dotenv_path='/config/.graphiti_env', override=True)
        
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
        self.embedding_dim = int(os.environ.get('EMBEDDING_DIM', '1536'))

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
        with open(f"../../datasets/{dataset_name}.json", "r") as f:
            dataset = json.load(f)
        
        if subset:
            return dataset[:subset]
        else:
            return dataset

    def get_corpus(self, dataset: Any, dataset_name: str) -> Dict[str, str]:
        """Get the corpus in episodes from the dataset."""
        if dataset_name == "2wikimultihopqa":
            passages = {}
            
            for datapoint in dataset:
                context = datapoint["context"]
                
                for passage in context:
                    title = passage[0].encode("utf-8").decode()
                    text = ""
                    for sen in passage[1]:
                        text.join(sen).encode("utf-8").decode()

                passages[title] = text
            
            return passages
        else:
            raise NotImplementedError(f"Dataset {dataset_name} not supported.")

    def get_queries(self, dataset: Any) -> List[str]:
        """Get the queries from the dataset."""
        queries: List[str] = []
        
        for datapoint in dataset:
            queries.append(
                str(
                    question=datapoint["question"].encode("utf-8").decode(),
                    answer=datapoint["answer"],
                    evidence=list(datapoint["supporting_facts"]),
                )
            )
        
        return queries

    async def create_graph(self, dataset_name: str, subset: int = 0):
        """Create the knowledge graph from the dataset."""
        print("Loading dataset...")
        dataset = self.load_dataset(dataset_name, subset)
        corpus = self.get_corpus(dataset, dataset_name)
        
        print(f"Dataset loaded. Corpus: {len(corpus)}")
        
        graphiti = self.setup_graphiti()
        
        try:
            # Initialize the graph db with graphiti's indices
            await graphiti.build_indices_and_constraints()
            
            # Add episodes to the graph
            for i, (title, text) in enumerate(tqdm(corpus.items(), desc="Adding episodes")):
                try:
                    start_time = time.perf_counter()
                    
                    await graphiti.add_episode(
                        name=f"{title}",
                        episode_body=text,
                        source=EpisodeType.text,
                        source_description='Wikipedia passage',
                        group_id=f"{dataset_name}_{subset}",
                        reference_time=datetime.now(timezone.utc),
                        entity_types=self.entity_types
                    )
                    
                    end_time = time.perf_counter()
                    execution_time = end_time - start_time
                    
                    self.logger.info(
                        f"[added_episode] Index: {i}, Title: {title}, "
                        f"Size: {len(text)}, Time: {execution_time:.6f}s"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Error adding episode {i} ({title}): {e}")
                    print(f"Error adding episode {i} ({title}): {e}")
            
            print("Graph creation completed!")
            
        finally:
            await graphiti.close()

    async def benchmark(self, dataset_name: str, subset: int = 0):
        """Benchmark the knowledge graph on queries."""
        print("Loading dataset...")
        dataset = self.load_dataset(dataset_name, subset)
        corpus = self.get_corpus(dataset, dataset_name)
        queries = self.get_queries(dataset)
        
        print(f"Dataset loaded. Queries: {len(queries)}")
        
        graphiti = self.setup_graphiti()
        results = []
        
        try:
            async def _query_task(query: Query) -> Dict[str, Any]:
                try:
                    start_time = time.perf_counter()
                    
                    # Search for relevant information
                    search_results = await graphiti.search(
                        query=query.question,
                        group_ids=[f"{dataset_name}_{subset}"],
                        limit=10
                    )
                    
                    end_time = time.perf_counter()
                    execution_time = end_time - start_time
                    
                    # Extract evidence from search results
                    evidence = []
                    response_facts = []
                    
                    if search_results:
                        for result in search_results:
                            response_facts.append(result.fact)
                            # Try to match back to original corpus titles
                            for hash_id, (title, text) in corpus.items():
                                if title in result.fact or any(word in result.fact.lower() for word in title.lower().split()):
                                    if title not in evidence:
                                        evidence.append(title)
                    
                    # Create a simple response from the facts
                    response = "\n".join(response_facts) if response_facts else "No relevant information found."
                    
                    self.logger.info(
                        f"[query] Question: '{query.question}', "
                        f"Time: {execution_time:.6f}s, Evidence count: {len(evidence)}"
                    )
                    
                    return {
                        "question": query.question,
                        "answer": response,
                        "evidence": evidence,
                        "ground_truth": [e[0] for e in query.evidence],
                        "execution_time": execution_time
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error processing query '{query.question}': {e}")
                    return {
                        "question": query.question,
                        "answer": f"Error: {str(e)}",
                        "evidence": [],
                        "ground_truth": [e[0] for e in query.evidence],
                        "execution_time": 0.0
                    }
            
            # Process all queries
            print("Processing queries...")
            tasks = [_query_task(query) for query in queries]
            
            for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Querying"):
                result = await task
                results.append(result)
            
            # Save results
            results_path = f"./results/graphiti/{dataset_name}_{subset}.json"
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)
            
            print(f"Benchmark completed! Results saved to {results_path}")
            
        finally:
            await graphiti.close()

    def compute_scores(self, dataset_name: str, subset: int = 0):
        """Compute and display benchmark scores."""
        results_path = f"./results/graphiti/{dataset_name}_{subset}.json"
        
        try:
            with open(results_path, "r") as f:
                answers = json.load(f)
        except FileNotFoundError:
            print(f"Results file not found: {results_path}")
            return
        
        try:
            with open(f"./questions/{dataset_name}_{subset}.json", "r") as f:
                questions_multihop = json.load(f)
        except FileNotFoundError:
            questions_multihop = []
        
        # Compute retrieval metrics
        retrieval_scores: List[float] = []
        retrieval_scores_multihop: List[float] = []
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
            
            if answer["question"] in questions_multihop:
                retrieval_scores_multihop.append(p_retrieved)
        
        # Print results
        print(f"\n=== Benchmark Results for {dataset_name} (subset: {subset}) ===")
        print(f"Total queries: {len(answers)}")
        print(f"Average retrieval score: {np.mean(retrieval_scores):.4f}")
        print(f"Percentage of queries with perfect retrieval: {np.mean([1 if s == 1.0 else 0 for s in retrieval_scores]):.4f}")
        print(f"Average execution time: {np.mean(execution_times):.4f}s")
        print(f"Total execution time: {np.sum(execution_times):.4f}s")
        
        if len(retrieval_scores_multihop):
            print(f"[multihop] Average retrieval score: {np.mean(retrieval_scores_multihop):.4f}")
            print(f"[multihop] Percentage of queries with perfect retrieval: {np.mean([1 if s == 1.0 else 0 for s in retrieval_scores_multihop]):.4f}")
        
        # Score distribution
        print(f"\nScore distribution:")
        for threshold in [0.0, 0.25, 0.5, 0.75, 1.0]:
            count = sum(1 for s in retrieval_scores if s >= threshold)
            print(f"  >= {threshold}: {count}/{len(retrieval_scores)} ({count/len(retrieval_scores):.4f})")


async def main():
    parser = argparse.ArgumentParser(description="Graphiti Benchmark CLI")
    parser.add_argument("-d", "--dataset", default="2wikimultihopqa", help="Dataset to use.")
    parser.add_argument("-n", type=int, default=0, help="Subset of corpus to use.")
    parser.add_argument("-c", "--create", action="store_true", help="Create the graph for the given dataset.")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Benchmark the graph for the given dataset.")
    parser.add_argument("-s", "--score", action="store_true", help="Report scores after benchmarking.")
    parser.add_argument("--working-dir", default="./db/graphiti", help="Working directory for graph storage.")
    
    args = parser.parse_args()
    
    benchmark = GraphitiBenchmark()
    
    if args.create:
        await benchmark.create_graph(args.dataset, args.n)
    
    if args.benchmark:
        await benchmark.benchmark(args.dataset, args.n)
    
    if args.benchmark or args.score:
        benchmark.compute_scores(args.dataset, args.n)


if __name__ == "__main__":
    asyncio.run(main())