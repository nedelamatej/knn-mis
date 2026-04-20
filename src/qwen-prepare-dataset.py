# [KNN] Konvolucni neuronove site
#
# Vysoke uceni technicke v Brne
# Fakulta informacnich technologii
#
# Nazev: qwen-prepare-dataset.py
# Autor: David Machu (xmachu05)
#        Matej Nedela (xnedel11)

from pathlib import Path
import argparse
import json
import random

parser = argparse.ArgumentParser(
  description='Prepare Qwen dataset',
  formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=30)
)

parser.add_argument('-i', '--input', help='path to input metadata JSON file', default='metadata.json')
parser.add_argument('-d', '--directory', help='directory to load JSON and JPG files from and save JSON files to', default='data')
parser.add_argument('-s', '--seed', help='random seed for shuffling the dataset', type=int, default=42)
parser.add_argument('-e', '--eval_ratio', help='proportion of the dataset to use for evaluation', type=float, default=0.1)
parser.add_argument('-t', '--test_ratio', help='proportion of the dataset to use for testing', type=float, default=0.1)

PROMPT = (
  '<image>\n'
  'Extract metadata from this title page of a scientific paper. Return only valid JSON in exactly '
  'the schema defined below.\n'
  '\n'
  'Schema:\n'
  '{\n'
  '  "title": "",\n'
  '  "authors": [\n'
  '    {\n'
  '      "firstName": "",\n'
  '      "lastName": "",\n'
  '      "email": null,\n'
  '      "institution": []\n'
  '    }\n'
  '  ],\n'
  '  "abstract": "",\n'
  '  "keywords": [],\n'
  '  "date": null\n'
  '}\n'
  '\n'
  'Rules:\n'
  '- Use only information visible on the page, do not invent missing values.\n'
  '- If email or date is missing, return `null`.\n'
  '- If institutions or keywords are missing, return `[]`.\n'
  '- If available, return date in ISO format (`YYYY-MM-DD`).\n'
  '- Preserve the title and author names exactly as written.\n'
  '- Return only raw JSON, do not wrap it in markdown formatting blocks.'
)

def convert_item(item, jpg_directory):
  article_id = item['id']

  image_name = f'{article_id}.jpg'
  image_path = jpg_directory / image_name

  if not image_path.exists():
    return None

  target = {
    'title': item['title'],
    'authors': item['authors'],
    'abstract': item['abstract'],
    'keywords': item['keywords'],
    'date': item['date'],
  }

  return {
    'id': article_id,
    'image': image_name,
    'conversations': [
      {
        'from': 'human',
        'value': PROMPT
      },
      {
        'from': 'gpt',
        'value': json.dumps(target, ensure_ascii=False)
      }
    ]
  }

def main():
  args = parser.parse_args()

  input_path = Path(args.directory, args.input)
  output_path = Path(args.directory)

  jpg_directory = Path(args.directory, 'jpg')

  with open(input_path, 'r', encoding='utf-8') as file:
    dataset = [convert_item(item, jpg_directory) for item in json.load(file)]
    dataset = [item for item in dataset if item is not None]

  random.seed(args.seed)
  random.shuffle(dataset)

  eval_size = round(len(dataset) * args.eval_ratio)
  test_size = round(len(dataset) * args.test_ratio)
  train_size = len(dataset) - eval_size - test_size

  train_data = dataset[:train_size]
  eval_data = dataset[train_size:train_size + eval_size]
  test_data = dataset[train_size + eval_size:]

  output_path.mkdir(parents=True, exist_ok=True)

  with open(output_path / 'train.json', 'w', encoding='utf-8') as file:
    json.dump(train_data, file, ensure_ascii=False, indent=2)

  with open(output_path / 'eval.json', 'w', encoding='utf-8') as file:
    json.dump(eval_data, file, ensure_ascii=False, indent=2)

  with open(output_path / 'test.json', 'w', encoding='utf-8') as file:
    json.dump(test_data, file, ensure_ascii=False, indent=2)

if __name__ == '__main__':
  main()
