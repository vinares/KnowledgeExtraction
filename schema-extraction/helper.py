
import json
from pathlib import Path
from models import *

def load_db(path):
    # Load the database from JSON
    input_file = Path(path)
    with input_file.open("r") as f:
        ds = json.load(f)
    
    return ds

def load_dialogues(path):
    ds = load_db(path)
    dialogues = []
    for item in ds:
        dialogues.append(item[0])
    return dialogues

def save_graph(result, path):
    # Save triplets to JSON
    output_file = Path("triplets.json")

    # Convert Pydantic models to dicts
    with output_file.open("w") as f:
        json.dump([triplet.dict() for triplet in result.relation_triplets], f, indent=2)