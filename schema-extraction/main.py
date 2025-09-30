import dspy
from models import *
from typing import List, Dict
from helper import *
import asyncio

async def process_episode(i, episode, extractor):
    episode_text = "\n".join(episode)
    try:
        # Run extractor in a thread to avoid blocking event loop
        triplets = await asyncio.to_thread(extractor, episode_text)
        if not triplets:
            print(f"[{i} ⚠️ No triplets extracted for dialogue {i+1}")
            return []
        print(f"{i} ✅ Found {len(triplets)} triplets in dialogue {i+1}")
        return triplets
    except Exception as e:
        print(f"{i} ❌ Extraction failed for dialogue {i+1}: {episode}\nerror:{e}")
        return []

async def process_episodes(dialogues, extractor):
    tasks = [
        process_episode(i, episode, extractor)
        for i, episode in enumerate(dialogues)
    ]
    results_per_episode = await asyncio.gather(*tasks)
    # Flatten results into one list
    all_results = [triplet for triplets in results_per_episode for triplet in triplets]
    return all_results

def main():
    # Configure the OpenAI LM
    # dspy.configure(lm=dspy.LM("openai/gpt-3.5-turbo"), api_key=OPEN)
    dspy.configure(lm=dspy.LM("openai/gpt-4-turbo"), api_key=OPEN)

    # Instantiate the predictor
    extractor = OptimizedRelationExtractor()
    # Example dialogue episode
    dialogues = load_dialogues(DEV_SET)

    results = asyncio.run(process_episodes(dialogues, extractor))
    # Save results
    save_graph(results, "./result/dev.json")


if __name__ == "__main__":
   main()