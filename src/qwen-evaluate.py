# [KNN] Konvolucni neuronove site
#
# Vysoke uceni technicke v Brne
# Fakulta informacnich technologii
#
# Nazev: evaluate-qwen.py
# Autor: David Machu (xmachu05)
#        Matej Nedela (xnedel11)

from pathlib import Path
from peft import PeftModel
from PIL import Image
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
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

parser.add_argument('-i', '--input', help='path to input data JSON file', default='test.json')
parser.add_argument('-o', '--output', help='path to save evaluation report JSON', default='report.json')
parser.add_argument('-c', '--count', help='number of articles to evaluate', type=int, default=None)
parser.add_argument('-m', '--model', help='path to the fine-tuned LoRA directory', default='qwen-lora-2026-04-11_09-20-37')
parser.add_argument('-d', '--directory', help='directory to load PDF and PNG files from', default='data')

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

def clean_output(text):
  text = text.strip()

  if text.startswith("```json"):
    text = text[7:]
  elif text.startswith("```"):
    text = text[3:]

  if text.endswith("```"):
    text = text[:-3]

  return text.strip()

def evaluate_model(model, processor, dataset, png_directory):
  rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

  results = {
    'time_total': 0,
    'time_average': 0,
    'total_samples': len(dataset),
    'valid_json_count': 0,
    'valid_json_ratio': 0,
    'accumulated_metrics': {
      'title_levenshtein': 0.0,
      'authors_f1': 0.0,
      'abstract_rouge': 0.0,
      'keywords_f1': 0.0,
    },
    'average_metrics': {
      'title_levenshtein': 0.0,
      'authors_f1': 0.0,
      'abstract_rouge': 0.0,
      'keywords_f1': 0.0,
    }
  }

  time_start = time.time()

  model.eval()

  for item in tqdm(dataset, ncols=80):
    image_path = png_directory / item['image']

    input_text = item['conversations'][0]['value']
    expected_output = json.loads(item['conversations'][1]['value'])

    message = [
      {
        'role': 'user',
        'content': [
          {'type': 'image', 'image': f'file://{image_path.absolute()}'},
          {'type': 'text', 'text': input_text.replace('<image>\n', '')}
        ]
      }
    ]

    text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    raw_image = Image.open(image_path).convert('RGB')

    inputs = processor(
      text=[text],
      images=[raw_image],
      padding=True,
      return_tensors='pt'
    ).to(model.device)

    with torch.no_grad():
      output_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
      output_ids = output_ids[:, inputs.input_ids.shape[1]:]

      output_text = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    try:
      output_json = json.loads(clean_output(output_text))

      results['valid_json_count'] += 1

      expected_title = expected_output.get('title', '')
      actual_title = output_json.get('title', '')
      results['accumulated_metrics']['title_levenshtein'] += calculate_levenshtein(expected_title, actual_title)

      expected_authors = expected_output.get('authors', [])
      actual_authors = output_json.get('authors', [])
      results['accumulated_metrics']['authors_f1'] += calculate_f1(expected_authors, actual_authors)

      expected_abstract = expected_output.get('abstract', '')
      actual_abstract = output_json.get('abstract', '')
      results['accumulated_metrics']['abstract_rouge'] += rouge.score(expected_abstract, actual_abstract)['rougeL'].fmeasure

      expected_keywords = expected_output.get('keywords', [])
      actual_keywords = output_json.get('keywords', [])
      results['accumulated_metrics']['keywords_f1'] += calculate_f1(expected_keywords, actual_keywords)
    except json.JSONDecodeError:
      pass

  results['time_total'] = time.time() - time_start
  results['time_average'] = results['time_total'] / len(dataset)
  results['valid_json_ratio'] = results['valid_json_count'] / len(dataset)

  if results['valid_json_count'] > 0:
    for key, accumulated in results['accumulated_metrics'].items():
      results['average_metrics'][key] = accumulated / results['valid_json_count']
  else:
    for key in results['accumulated_metrics'].keys():
      results['average_metrics'][key] = 0.0

  return results

def main():
  args = parser.parse_args()

  input_path = Path(args.input)
  output_path = Path(args.output)

  png_directory = Path(args.directory, 'png')

  processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct')

  with open(input_path, 'r', encoding='utf-8') as file:
    dataset = json.load(file)

  if args.count:
    dataset = dataset[:args.count]

  base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2.5-VL-3B-Instruct',
    torch_dtype=torch.float16,
    device_map="auto"
  )

  base_results = evaluate_model(base_model, processor, dataset, png_directory)

  # clear memory before loading tuned model
  del base_model; torch.cuda.empty_cache(); gc.collect()

  base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2.5-VL-3B-Instruct',
    torch_dtype=torch.float16,
    device_map="auto"
  )

  tuned_model = PeftModel.from_pretrained(base_model, args.model)
  tuned_results = evaluate_model(tuned_model, processor, dataset, png_directory)

  report = {
    'base_model': base_results,
    'tuned_model': tuned_results
  }

  with open(output_path, 'w', encoding='utf-8') as file:
    json.dump(report, file, ensure_ascii=False, indent=2)

if __name__ == '__main__':
  main()
