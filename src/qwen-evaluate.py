# [KNN] Konvolucni neuronove site
#
# Vysoke uceni technicke v Brne
# Fakulta informacnich technologii
#
# Nazev: qwen-evaluate.py
# Autor: David Machu (xmachu05)
#        Matej Nedela (xnedel11)

from pathlib import Path
from peft import PeftModel
from PIL import Image
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
import argparse
import editdistance
import gc
import json
import time
import torch

parser = argparse.ArgumentParser(
  description='Evaluate and compare Base Qwen vs Fine-tuned Qwen',
  formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=30)
)

parser.add_argument('-m', '--model', help='name of the base model to evaluate', default='Qwen2.5-VL-3B-Instruct')
parser.add_argument('-l', '--lora', help='path to the fine-tuned LoRA directory', default='Qwen2.5-VL-3B-Instruct-lora-19611240')
parser.add_argument('-c', '--count', help='number of articles to evaluate', type=int, default=None)
parser.add_argument('-b', '--batch-size', help='number of articles to process per forward pass', type=int, default=4)
parser.add_argument('-i', '--input', help='path to input data JSON file', default='test.json')
parser.add_argument('-o', '--output', help='path to save evaluation report JSON', default='report.json')
parser.add_argument('-d', '--directory', help='directory to load JSON and JPG files from', default='data')

def calculate_f1(expected_list, actual_list):
  if not isinstance(expected_list, list): expected_list = []
  if not isinstance(actual_list, list): actual_list = []

  expected_set = set([str(x).lower().strip() for x in expected_list])
  actual_set = set([str(x).lower().strip() for x in actual_list])

  if not expected_set and not actual_set: return 1.0
  if not expected_set or not actual_set: return 0.0

  true_positives = len(expected_set.intersection(actual_set))
  false_positives = len(actual_set - expected_set)
  false_negatives = len(expected_set - actual_set)

  if true_positives == 0: return 0.0

  precision = true_positives / (true_positives + false_positives)
  recall = true_positives / (true_positives + false_negatives)

  return 2 * (precision * recall) / (precision + recall)

def calculate_levenshtein(expected_text, actual_text):
  expected_text = str(expected_text).strip().lower()
  actual_text = str(actual_text).strip().lower()

  if not expected_text and not actual_text: return 1.0
  if not expected_text or not actual_text: return 0.0

  distance = editdistance.eval(expected_text, actual_text)
  max_len = max(len(expected_text), len(actual_text))

  return 1.0 - (distance / max_len)

def calculate_author_metrics(expected_authors, actual_authors):
  if not isinstance(expected_authors, list): expected_authors = []
  if not isinstance(actual_authors, list): actual_authors = []

  if not expected_authors and not actual_authors: return {'firstName': 1.0, 'lastName': 1.0, 'email': 1.0, 'institution': 1.0}
  if not expected_authors or not actual_authors: return {'firstName': 0.0, 'lastName': 0.0, 'email': 0.0, 'institution': 0.0}

  n = max(len(expected_authors), len(actual_authors))

  totals = {'firstName': 0.0, 'lastName': 0.0, 'email': 0.0, 'institution': 0.0}

  for i in range(n):
    expected_author = expected_authors[i] if i < len(expected_authors) else {}
    actual_author = actual_authors[i] if i < len(actual_authors) else {}

    if not isinstance(expected_author, dict): expected_author = {}
    if not isinstance(actual_author, dict): actual_author = {}

    for field in ('firstName', 'lastName', 'email'):
      expected_value = expected_author.get(field) or ''
      actual_value = actual_author.get(field) or ''

      if not expected_value and field == 'email':
        totals[field] += 1.0
      else:
        totals[field] += calculate_levenshtein(expected_value, actual_value)

    expected_institution = expected_author.get('institution') or []
    actual_institution = actual_author.get('institution') or []

    if not expected_institution:
      totals['institution'] += 1.0
    else:
      totals['institution'] += calculate_f1(expected_institution, actual_institution)

  return {k: v / n for k, v in totals.items()}

def clean_output(text):
  text = text.strip()

  if text.startswith("```json"):
    text = text[7:]
  elif text.startswith("```"):
    text = text[3:]

  if text.endswith("```"):
    text = text[:-3]

  return text.strip()

def evaluate_model(model, processor, dataset, jpg_directory, batch_size):
  rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

  results = {
    'time_total': 0,
    'time_average': 0,
    'total_samples': len(dataset),
    'valid_json_count': 0,
    'valid_json_ratio': 0,
    'accumulated_metrics': {
      'title_levenshtein': 0.0,
      'authors_firstName_levenshtein': 0.0,
      'authors_lastName_levenshtein': 0.0,
      'authors_email_levenshtein': 0.0,
      'authors_institution_f1': 0.0,
      'abstract_rouge': 0.0,
      'keywords_f1': 0.0,
      'date_exact': 0.0,
    },
    'average_metrics': {
      'title_levenshtein': 0.0,
      'authors_firstName_levenshtein': 0.0,
      'authors_lastName_levenshtein': 0.0,
      'authors_email_levenshtein': 0.0,
      'authors_institution_f1': 0.0,
      'abstract_rouge': 0.0,
      'keywords_f1': 0.0,
      'date_exact': 0.0,
    },
    'outputs': [],
  }

  time_start = time.time()

  model.eval()
  processor.tokenizer.padding_side = 'left'

  for batch_start in tqdm(range(0, len(dataset), batch_size), ncols=80):
    batch = dataset[batch_start:batch_start + batch_size]

    texts = []
    raw_images = []
    batch_meta = []

    for item in batch:
      image_path = jpg_directory / item['image']
      input_text = item['conversations'][0]['value']
      expected_text = item['conversations'][1]['value']

      message = [
        {
          'role': 'user',
          'content': [
            {'type': 'image', 'image': f'file://{image_path.absolute()}'},
            {'type': 'text', 'text': input_text.replace('<image>\n', '')}
          ]
        }
      ]

      texts.append(processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True))
      raw_images.append(Image.open(image_path).convert('RGB'))
      batch_meta.append({'image': item['image'], 'input': input_text, 'expected': expected_text})

    inputs = processor(
      text=texts,
      images=raw_images,
      padding=True,
      return_tensors='pt'
    ).to(model.device)

    with torch.no_grad():
      output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
      output_ids = output_ids[:, inputs.input_ids.shape[1]:]
      output_texts = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    for meta, output_text in zip(batch_meta, output_texts):
      expected_output = json.loads(meta['expected'])

      item_result = {
        'image': meta['image'],
        'valid_json': False,
        'expected': expected_output,
        'actual': None,
        'metrics': None,
      }

      try:
        output_json = json.loads(clean_output(output_text))

        results['valid_json_count'] += 1
        item_result['valid_json'] = True
        item_result['actual'] = output_json
      except json.JSONDecodeError:
        output_json = None

      if output_json is not None:
        expected_title = expected_output.get('title') or ''
        actual_title = output_json.get('title') or ''
        title_levenshtein = calculate_levenshtein(expected_title, actual_title)

        expected_authors = expected_output.get('authors') or []
        actual_authors = output_json.get('authors') or []
        author_metrics = calculate_author_metrics(expected_authors, actual_authors)

        expected_abstract = expected_output.get('abstract') or ''
        actual_abstract = output_json.get('abstract') or ''
        abstract_rouge = rouge.score(expected_abstract, actual_abstract)['rougeL'].fmeasure

        expected_keywords = expected_output.get('keywords') or []
        actual_keywords = output_json.get('keywords') or []
        keywords_f1 = 1.0 if not expected_keywords else calculate_f1(expected_keywords, actual_keywords)

        expected_date = str(expected_output.get('date') or '').strip()
        actual_date = str(output_json.get('date') or '').strip()
        date_exact = 1.0 if expected_date == actual_date else 0.0

        item_result['metrics'] = {
          'title_levenshtein': title_levenshtein,
          'authors_firstName_levenshtein': author_metrics['firstName'],
          'authors_lastName_levenshtein': author_metrics['lastName'],
          'authors_email_levenshtein': author_metrics['email'],
          'authors_institution_f1': author_metrics['institution'],
          'abstract_rouge': abstract_rouge,
          'keywords_f1': keywords_f1,
          'date_exact': date_exact,
        }
      else:
        item_result['metrics'] = {
          'title_levenshtein': 0.0,
          'authors_firstName_levenshtein': 0.0,
          'authors_lastName_levenshtein': 0.0,
          'authors_email_levenshtein': 0.0,
          'authors_institution_f1': 0.0,
          'abstract_rouge': 0.0,
          'keywords_f1': 0.0,
          'date_exact': 0.0,
        }

      for key, value in item_result['metrics'].items():
        results['accumulated_metrics'][key] += value

      results['outputs'].append(item_result)

  results['time_total'] = time.time() - time_start
  results['time_average'] = results['time_total'] / len(dataset)
  results['valid_json_ratio'] = results['valid_json_count'] / len(dataset)

  for key, accumulated in results['accumulated_metrics'].items():
    results['average_metrics'][key] = accumulated / results['total_samples']

  return {
    'summary': {
      'time_total': results['time_total'],
      'time_average': results['time_average'],
      'total_samples': results['total_samples'],
      'valid_json_count': results['valid_json_count'],
      'valid_json_ratio': results['valid_json_ratio'],
    },
    'metrics': results['average_metrics'],
    'samples': results['outputs'],
  }

def main():
  args = parser.parse_args()

  input_path = Path(args.directory, args.input)
  output_path = Path(args.output)

  jpg_directory = Path(args.directory, 'jpg')

  processor = AutoProcessor.from_pretrained(
    f'Qwen/{args.model}',
    min_pixels=256 * 28 * 28,
    max_pixels=1280 * 28 * 28,
    use_fast=False
  )

  with open(input_path, 'r', encoding='utf-8') as file:
    dataset = json.load(file)

  if args.count:
    dataset = dataset[:args.count]

  base_model = AutoModelForImageTextToText.from_pretrained(
    f'Qwen/{args.model}',
    dtype=torch.float16,
    device_map='auto'
  )

  base_results = evaluate_model(base_model, processor, dataset, jpg_directory, args.batch_size)

  # clear memory before loading tuned model
  del base_model; torch.cuda.empty_cache(); gc.collect()

  base_model = AutoModelForImageTextToText.from_pretrained(
    f'Qwen/{args.model}',
    dtype=torch.float16,
    device_map='auto'
  )

  tuned_model = PeftModel.from_pretrained(base_model, args.lora)
  tuned_results = evaluate_model(tuned_model, processor, dataset, jpg_directory, args.batch_size)

  samples = []

  for base_item, tuned_item in zip(base_results['samples'], tuned_results['samples']):
    samples.append({
      'image': base_item['image'],
      'valid_json': {'base': base_item['valid_json'], 'tuned': tuned_item['valid_json']},
      'metrics': {'base': base_item['metrics'], 'tuned': tuned_item['metrics']},
      'expected': base_item['expected'],
      'base': base_item['actual'],
      'tuned': tuned_item['actual'],
    })

  report = {
    'metrics': {'base': base_results['metrics'], 'tuned': tuned_results['metrics']},
    'summary': {'base': base_results['summary'], 'tuned': tuned_results['summary']},
    'samples': samples,
  }

  with open(output_path, 'w', encoding='utf-8') as file:
    json.dump(report, file, ensure_ascii=False, indent=2)

if __name__ == '__main__':
  main()
