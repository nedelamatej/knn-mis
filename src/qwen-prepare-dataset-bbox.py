# [KNN] Konvolucni neuronove site
#
# Vysoke uceni technicke v Brne
# Fakulta informacnich technologii
#
# Nazev: qwen-prepare-dataset-bbox.py
# Autor: David Machu (xmachu05)
#        Matej Nedela (xnedel11)

from pathlib import Path
import argparse
import json
import random

from alto_utils import (
  parse_alto_words,
  find_text_bbox,
  find_title_bbox,
  find_author_bboxes,
)

parser = argparse.ArgumentParser(
  description='Prepare Qwen dataset with bounding boxes',
  formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=30)
)

parser.add_argument('-i', '--input', help='path to input metadata JSON file', default='metadata.json')
parser.add_argument('-d', '--directory', help='directory to load JSON, JPG and ALTO files from and save JSON files to', default='data')
parser.add_argument('-s', '--seed', help='random seed for shuffling the dataset', type=int, default=42)
parser.add_argument('-e', '--eval_ratio', help='proportion of the dataset to use for evaluation', type=float, default=0.1)
parser.add_argument('-t', '--test_ratio', help='proportion of the dataset to use for testing', type=float, default=0.1)

PROMPT = (
  '<image>\n'
  'Extract metadata from this title page of a scientific paper. Return only valid JSON in exactly '
  'the schema defined below. Include bounding boxes for extracted values.\n'
  '\n'
  'Bounding box format:\n'
  '- bbox is [x1, y1, x2, y2] in image pixel coordinates.\n'
  '- If the value is missing, use null for the value and null for bbox.\n'
  '- If the value exists but its position cannot be determined, use null for bbox.\n'
  '\n'
  'Schema:\n'
  '{\n'
  '  "title": {"text": "", "bbox": null},\n'
  '  "authors": [\n'
  '    {\n'
  '      "firstName": "",\n'
  '      "lastName": "",\n'
  '      "email": null,\n'
  '      "institution": [],\n'
  '      "bbox": null\n'
  '    }\n'
  '  ],\n'
  '  "abstract": {"text": "", "bbox": null},\n'
  '  "keywords": [\n'
  '    {"text": "", "bbox": null}\n'
  '  ],\n'
  '  "date": {"text": null, "bbox": null}\n'
  '}\n'
  '\n'
  'Rules:\n'
  '- Use only information visible on the page, do not invent missing values.\n'
  '- If email or date is missing, return null.\n'
  '- If institutions or keywords are missing, return [].\n'
  '- If a bbox cannot be determined reliably, return null for bbox.\n'
  '- If available, return date in ISO format YYYY-MM-DD.\n'
  '- Preserve the title and author names exactly as written.\n'
  '- Return only raw JSON, do not wrap it in markdown formatting blocks.'
)


def convert_item(item, jpg_directory, alto_directory):
  article_id = item['id']

  image_name = f'{article_id}.jpg'
  image_path = jpg_directory / image_name
  alto_path = alto_directory / f'{article_id}.xml'

  if not image_path.exists():
    return None

  if not alto_path.exists():
    return None

  words = parse_alto_words(alto_path)

  abstract_bbox = find_text_bbox(item.get('abstract'), words)
  title_bbox = find_title_bbox(item.get('title'), words, abstract_bbox)

  source_authors = item.get('authors', [])
  author_bboxes = find_author_bboxes(source_authors, words, title_bbox, abstract_bbox)

  authors = []
  for author, author_bbox in zip(source_authors, author_bboxes):
    authors.append({
      'firstName': author.get('firstName', ''),
      'lastName': author.get('lastName', ''),
      'email': author.get('email'),
      'institution': author.get('institution', []),
      'bbox': author_bbox,
    })

  keywords = []
  for keyword in item.get('keywords', []):
    keywords.append({
      'text': keyword,
      'bbox': find_text_bbox(keyword, words),
    })

  target = {
    'title': {
      'text': item.get('title', ''),
      'bbox': title_bbox,
    },
    'authors': authors,
    'abstract': {
      'text': item.get('abstract', ''),
      'bbox': abstract_bbox,
    },
    'keywords': keywords,
    'date': {
      'text': item.get('date'),
      'bbox': None,
    },
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
  alto_directory = Path(args.directory, 'alto')

  with open(input_path, 'r', encoding='utf-8') as file:
    dataset = [convert_item(item, jpg_directory, alto_directory) for item in json.load(file)]
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

  with open(output_path / 'train_bbox.json', 'w', encoding='utf-8') as file:
    json.dump(train_data, file, ensure_ascii=False, indent=2)

  with open(output_path / 'eval_bbox.json', 'w', encoding='utf-8') as file:
    json.dump(eval_data, file, ensure_ascii=False, indent=2)

  with open(output_path / 'test_bbox.json', 'w', encoding='utf-8') as file:
    json.dump(test_data, file, ensure_ascii=False, indent=2)

  print(f'Total: {len(dataset)}')
  print(f'Train: {len(train_data)}')
  print(f'Eval: {len(eval_data)}')
  print(f'Test: {len(test_data)}')

if __name__ == '__main__':
  main()
