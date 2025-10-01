RAW_RELATIONS_FILE = "result/dev.txt"
RAW_DEV_RELATIONS_JSON = "result/dev.json"
RAW_DEV_RELATIONS_FILE = "result/dev.txt"

CLEANSED_RELATIONS_FILE = "result/cleansed.txt"
GROUND_TRUTH_FILE = "../data/fandom_triples.txt"
GROUND_TRUTH_MAPPING_FILE = "../data/matching_table.txt"
REDUCED_GROUND_TRUTH_FILE = "../data/reduced_ground_truth.txt"

from collections import defaultdict
from models import *

def extract_relations_from_fandom_triples():
    relations = defaultdict(list)
    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("||")
            if len(parts) == 3:
                key = (parts[0], parts[1])
                relations[key].append(parts[2])
    for key in relations:
        relations[key] = list(set(relations[key]))  # Remove duplicates
    return relations

def get_mapping_table():
    mapping = dict()
    with open(GROUND_TRUTH_MAPPING_FILE, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("||")
            if len(parts) == 2:
                mapping[parts[0]] = parts[1]
    return mapping

def reduce_relations(relations, mapping):
    reduced = defaultdict(set)
    for (x, y), rels in relations.items():
        for r in rels:
            mapped_r = mapping.get(r, None)
            if mapped_r and mapped_r in ALL_RELATION_TYPES:
                reduced[(x, y)].add(mapped_r)
    # Convert sets back to lists
    return {key: list(vals) for key, vals in reduced.items()}

def generate_source_of_truth():
    relations = extract_relations_from_fandom_triples()
    mapping = get_mapping_table()
    reduced_relations = reduce_relations(relations, mapping)
    
    with open(REDUCED_GROUND_TRUTH_FILE, "w", encoding="utf-8") as f:
        for (x, y), rels in reduced_relations.items():
            for r in rels:
                f.write(f"{x}||{y}||{r}\n")
    print(f"✅ Generated reduced ground truth at {REDUCED_GROUND_TRUTH_FILE}")

def load_relations(file_path):
    relations = defaultdict(list)
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("||")
            if len(parts) == 3:
                key = (parts[0], parts[1])
                relations[key].append(parts[2])
    return relations

def main():
    # generate_source_of_truth()

    ground_truth = load_relations(REDUCED_GROUND_TRUTH_FILE)
    raw_relations = load_relations(RAW_RELATIONS_FILE)
    
    print(f"Ground truth relations: {len(ground_truth)}")
    print(f"Raw extracted relations: {len(raw_relations)}")

if __name__ == "__main__":
    main()