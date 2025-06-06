import asyncio
from src.graphiti.benchmark import GraphitiBenchmark

# Specify a subset of the dataset to use
DATASET_NAME = "2wikimultihopqa"
SUBSET = (0, 10)

async def main():
    benchmark = GraphitiBenchmark()
    
    # await benchmark.create_graph(dataset_name=DATASET_NAME, subset=SUBSET)
    await benchmark.benchmark(dataset_name=DATASET_NAME, subset=SUBSET)
    benchmark.compute_scores(dataset_name=DATASET_NAME)


if __name__ == "__main__":
    asyncio.run(main())