import dspy
import sys
import os

# Add parent folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schema_extraction import models
from graphdb import query

def main():
    dspy.configure(lm=dspy.LM("gpt-4.1-mini"), api_key=models.OPEN)

    agent = models.QuestionAnswerAgent()
    question = "What's the relation between emma and joshua?"
    relations = query.start_sync("Emma", "Joshua")
    print(relations)
    answer = agent(question_text=question, entity_relations=relations)
    print(f"Q: {question}\nA: {answer}")

if __name__ == "__main__":
   main()
