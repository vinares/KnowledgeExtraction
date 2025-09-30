import json

def json_to_txt(json_file, txt_file):
    # Load JSON data (list of dicts)
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Write to text file in desired format
    with open(txt_file, "w", encoding="utf-8") as f:
        for entry in data:
            line = f"{entry['x']}||{entry['y']}||{entry['r']}"
            f.write(line + "\n")

if __name__ == "__main__":
    # Example usage
    json_to_txt("result/dev.json", "result/dev.txt")
    print("✅ Conversion complete: relations.txt")
