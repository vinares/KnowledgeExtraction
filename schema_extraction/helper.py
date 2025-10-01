import asyncio
import json
from pathlib import Path
from models import *


def load_db(path):
    # Load the database from JSON
    input_file = Path(path)
    with input_file.open("r") as f:
        ds = json.load(f)
    
    return ds

def load_dialogues(path, count=None):
    ds = load_db(path)
    count = min(count, len(ds)) if count is not None else len(ds)
    dialogues = []
    for item in ds[:count]:
        dialogues.append(item[0])
    return dialogues

def load_relations(path, count=None):
    ds = load_db(path)
    count = min(count, len(ds)) if count is not None else len(ds)
    relations = []
    for item in ds[:count]:
        relations.append(item[1])
    return relations

def save_graph(results, path):
    output_file = Path(path)
    # Ensure parent directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    all_triplets = [triplet.dict() for triplet in results]

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(all_triplets, f, indent=2, ensure_ascii=False)
