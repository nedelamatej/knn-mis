# [KNN] Konvolucni neuronove site
#
# Vysoke uceni technicke v Brne
# Fakulta informacnich technologii
#
# Nazev: make_small_qwen_dataset.py
# Autor: David Machu (xmachu05)
#        Matej Nedela (xnedel11)

from pathlib import Path
import argparse
import json
import random

parser = argparse.ArgumentParser(
  description='Create smaller train/eval/test JSON datasets for quick experiments.',
  formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=30)
)

parser.add_argument('--train-input', help='path to input train JSON file', default='train.json')
parser.add_argument('--eval-input', help='path to input eval JSON file', default='eval.json')
parser.add_argument('--test-input', help='path to input test JSON file', default='test.json')

parser.add_argument('--train-output', help='path to output small train JSON file', default='train_small.json')
parser.add_argument('--eval-output', help='path to output small eval JSON file', default='eval_small.json')
parser.add_argument('--test-output', help='path to output small test JSON file', default='test_small.json')

parser.add_argument('--train-count', help='number of records in small train dataset', type=int, default=200)
parser.add_argument('--eval-count', help='number of records in small eval dataset', type=int, default=50)
parser.add_argument('--test-count', help='number of records in small test dataset', type=int, default=50)

parser.add_argument('--seed', help='random seed for reproducible sampling', type=int, default=42)

args = parser.parse_args()

train_input_path = Path(args.train_input)
eval_input_path = Path(args.eval_input)
test_input_path = Path(args.test_input)

train_output_path = Path(args.train_output)
eval_output_path = Path(args.eval_output)
test_output_path = Path(args.test_output)

def load_json(path: Path):
  with open(path, 'r', encoding='utf-8') as file:
    return json.load(file)

def save_json(path: Path, data):
  with open(path, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=2)

def sample_records(data, count: int, rng: random.Random, dataset_name: str):
  if count < 0:
    raise ValueError(f'{dataset_name}: count must be non-negative')

  if count > len(data):
    raise ValueError(
      f'{dataset_name}: requested {count} records, but dataset contains only {len(data)} records'
    )

  sampled = list(data)
  rng.shuffle(sampled)
  return sampled[:count]

train_data = load_json(train_input_path)
eval_data = load_json(eval_input_path)
test_data = load_json(test_input_path)

rng = random.Random(args.seed)

train_small = sample_records(train_data, args.train_count, rng, 'train')
eval_small = sample_records(eval_data, args.eval_count, rng, 'eval')
test_small = sample_records(test_data, args.test_count, rng, 'test')

save_json(train_output_path, train_small)
save_json(eval_output_path, eval_small)
save_json(test_output_path, test_small)

print(f'Loaded train: {len(train_data)}')
print(f'Loaded eval:  {len(eval_data)}')
print(f'Loaded test:  {len(test_data)}')

print(f'Saved {len(train_small)} samples to {train_output_path}')
print(f'Saved {len(eval_small)} samples to {eval_output_path}')
print(f'Saved {len(test_small)} samples to {test_output_path}')
