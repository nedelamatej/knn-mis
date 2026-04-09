# [KNN] Konvolucni neuronove site
#
# Vysoke uceni technicke v Brne
# Fakulta informacnich technologii
#
# Nazev: prepare_qwen_dataset.py
# Autor: David Machu (xmachu05)
#        Matej Nedela (xnedel11)

import json
import random
from pathlib import Path

INPUT_METADATA = "data/metadata.json"
IMAGE_DIR = "data/png"

TRAIN_OUT = "train.json"
EVAL_OUT = "eval.json"
TEST_OUT = "test.json"

EVAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42

PROMPT = (
    "<image>\n"
    "Extract metadata from this first page of a scientific paper.\n"
    "Return only valid JSON in exactly this schema:\n"
    "{\n"
    '  "title": "",\n'
    '  "authors": [],\n'
    '  "abstract": "",\n'
    '  "keywords": []\n'
    "}\n\n"
    "Rules:\n"
    "- Use only information visible on the page.\n"
    "- Do not invent missing values.\n"
    "- If keywords are missing, return [].\n"
    "- Preserve the title and author names exactly as written.\n"
    "- Return only JSON."
)

def author_to_string(author_obj):
    first = (author_obj.get("firstName") or "").strip()
    last = (author_obj.get("lastName") or "").strip()
    full = f"{first} {last}".strip()
    return " ".join(full.split())

def convert_record(item):
    paper_id = item["id"]
    image_name = f"{paper_id}.png"

    image_path = Path(IMAGE_DIR) / image_name
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")

    authors = [author_to_string(a) for a in item.get("authors", [])]
    keywords = item.get("keywords")
    if keywords is None:
        keywords = []

    target = {
        "title": item.get("title", "") or "",
        "authors": authors,
        "abstract": item.get("abstract", "") or "",
        "keywords": keywords,
    }

    return {
        "id": paper_id,
        "image": image_name,
        "conversations": [
            {
                "from": "human",
                "value": PROMPT
            },
            {
                "from": "gpt",
                "value": json.dumps(target, ensure_ascii=False)
            }
        ]
    }

def main():
    with open(INPUT_METADATA, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    converted = [convert_record(item) for item in metadata]

    random.seed(SEED)
    random.shuffle(converted)

    total = len(converted)
    eval_size = int(total * EVAL_RATIO)
    test_size = int(total * TEST_RATIO)

    eval_data = converted[:eval_size]
    test_data = converted[eval_size:eval_size + test_size]
    train_data = converted[eval_size + test_size:]

    with open(TRAIN_OUT, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(EVAL_OUT, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)

    with open(TEST_OUT, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(train_data)} samples to {TRAIN_OUT}")
    print(f"Saved {len(eval_data)} samples to {EVAL_OUT}")
    print(f"Saved {len(test_data)} samples to {TEST_OUT}")

if __name__ == "__main__":
    main()
    