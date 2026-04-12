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
parser.add_argument('-o', '--output', help='directory to save JSON files to', default='.')
parser.add_argument('-d', '--directory', help='directory to load PDF and PNG files from', default='data')
parser.add_argument('-s', '--seed', help='random seed for shuffling the dataset', type=int, default=42)
parser.add_argument('-e', '--eval_ratio', help='proportion of the dataset to use for evaluation', type=float, default=0.1)
parser.add_argument('-t', '--test_ratio', help='proportion of the dataset to use for testing', type=float, default=0.1)

PROMPT = (
  '<image>\n'
  'Extract metadata from this first page of a scientific paper.\n'
  'Return only valid JSON in exactly this schema:\n'
  '{\n'
  '  "title": "",\n'
  '  "authors": [],\n'
  '  "abstract": "",\n'
  '  "keywords": []\n'
  '}\n'
  '\n'
  'Rules:\n'
  '- Use only information visible on the page.\n'
  '- Do not invent missing values.\n'
  '- If keywords are missing, return [].\n'
  '- Preserve the title and author names exactly as written.\n'
  '- Return only JSON.'
)

def convert_author(author):
  first = (author.get('firstName') or '').strip()
  last = (author.get('lastName') or '').strip()

  return ' '.join(f'{first} {last}'.split())

def convert_item(item, png_directory):
  article_id = item['id']

  image_name = f'{article_id}.png'
  image_path = png_directory / image_name

  if not image_path.exists():
    print(f'Warning: Skipping missing image -> {image_path}')

    return None

  target = {
    'title': item.get('title', '') or '',
    'authors': [convert_author(author) for author in item.get('authors', [])],
    'abstract': item.get('abstract', '') or '',
    'keywords': item.get('keywords', []) or [],
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

  input_path = Path(args.input)
  output_path = Path(args.output)

  png_directory = Path(args.directory, 'png')

  with open(input_path, 'r', encoding='utf-8') as file:
    dataset = [convert_item(item, png_directory) for item in json.load(file)]
    dataset = [item for item in dataset if item is not None]

  random.seed(args.seed)
  random.shuffle(dataset)

  eval_size = int(len(dataset) * args.eval_ratio)
  test_size = int(len(dataset) * args.test_ratio)

  train_data = dataset[eval_size + test_size:]
  eval_data = dataset[:eval_size]
  test_data = dataset[eval_size:eval_size + test_size]

  output_path.mkdir(parents=True, exist_ok=True)

  with open(output_path / 'train.json', 'w', encoding='utf-8') as file:
    json.dump(train_data, file, ensure_ascii=False, indent=2)

  with open(output_path / 'eval.json', 'w', encoding='utf-8') as file:
    json.dump(eval_data, file, ensure_ascii=False, indent=2)

  with open(output_path / 'test.json', 'w', encoding='utf-8') as file:
    json.dump(test_data, file, ensure_ascii=False, indent=2)

  print(f'Saved {len(train_data)} samples to `{output_path / "train.json"}`.')
  print(f'Saved {len(eval_data)} samples to `{output_path / "eval.json"}`.')
  print(f'Saved {len(test_data)} samples to `{output_path / "test.json"}`.')

if __name__ == '__main__':
  main()
