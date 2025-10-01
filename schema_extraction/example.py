import dspy
from models import *
from typing import List, Dict
from helper import *

def main():
   # Configure the OpenAI LM
   dspy.configure(lm=dspy.LM("gpt-4.1-mini"), api_key=OPEN)

   # Instantiate the predictor
   extractor = OptimizedRelationExtractor()
   # Example dialogue episode
   episode = [
      "Speaker 1: Hey Pheebs.",
      "Speaker 2: Hey!",
      "Speaker 1: Any sign of your brother?",
      "Speaker 2: No, but he's always late.",
      "Speaker 1: I thought you only met him once?",
      "Speaker 2: Yeah, I did. I think it sounds y'know big sistery, y'know, 'Frank's always late.'",
      "Speaker 1: Well relax, he'll be here."
   ]
   
   # Process the entire episode at once
   triplets = extractor("\n".join(episode))
   print(f"all entity types: {ALL_ENTITY_TYPES}")
   print(f"all relation types: {ALL_RELATION_TYPES}")
   print("Extracted Triplets from Entire Episode:")
   for triplet in triplets:
      print(triplet)

if __name__ == "__main__":
   main()