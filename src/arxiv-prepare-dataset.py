# [KNN] Konvolucni neuronove site
#
# Vysoke uceni technicke v Brne
# Fakulta informacnich technologii
#
# Nazev: arxiv-prepare-dataset.py
# Autor: David Machu (xmachu05)
#        Matej Nedela (xnedel11)

from datetime import datetime
from itertools import islice
from pathlib import Path
from tqdm import tqdm
import argparse
import json
import pymupdf
import re
import requests
import time

pymupdf.TOOLS.mupdf_display_errors(False)

parser = argparse.ArgumentParser(
  description='Prepare arXiv dataset - download articles as PDFs, save them as PNGs and parse their metadata into a JSON file.',
  epilog='Input metadata JSON file can be downloaded from "https://www.kaggle.com/datasets/Cornell-University/arxiv" (~5 GB).',
  formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=30)
)

parser.add_argument('-s', '--start', help='start index for processing articles', type=int, default=0)
parser.add_argument('-c', '--count', help='number of articles to process', type=int, default=20)
parser.add_argument('-i', '--input', help='path to input metadata JSON file', default='arxiv-metadata-snapshot.json')
parser.add_argument('-o', '--output', help='path to output metadata JSON file', default='metadata.json')
parser.add_argument('-d', '--directory', help='directory to save PDF and PNG files to', default='data')
parser.add_argument('--dpi', help='DPI resolution for the output PNG images', type=int, default=150)

def main():
  args = parser.parse_args()

  input_path = Path(args.input)
  output_path = Path(args.output)

  pdf_directory = Path(args.directory, 'pdf')
  png_directory = Path(args.directory, 'png')

  pdf_directory.mkdir(parents=True, exist_ok=True)
  png_directory.mkdir(parents=True, exist_ok=True)

  data = []

  last_request_time = 0

  session = requests.Session()

  with open(input_path, 'r', encoding='utf-8') as file:
    for idx, line in tqdm(islice(enumerate(file), args.start, args.start + args.count), total=args.count, ncols=80):
      article_id = None

      try:
        article = json.loads(line)
        article_id = article['id'].replace('/', '_')

        pdf_path = pdf_directory / f'{article_id}.pdf'
        png_path = png_directory / f'{article_id}.png'

        if not pdf_path.is_file():
          time.sleep(max(0, 3 - (time.monotonic() - last_request_time)))

          last_request_time = time.monotonic()

          response = session.get(f"https://arxiv.org/pdf/{article['id']}", timeout=30)
          response.raise_for_status()

          with open(pdf_path, 'wb') as pdf_file:
            pdf_file.write(response.content)

        with pymupdf.open(pdf_path) as pdf_file:
          pdf_file.load_page(0).get_pixmap(dpi=args.dpi).save(png_path)

        data.append({
          'idx': idx,
          'id': article_id,
          'title': re.sub(r'\s+', ' ', article.get('title', '')).strip(),
          'authors': [
            {
              'firstName': author[1] if len(author) > 1 else None,
              'lastName': author[0] if len(author) > 0 else None,
              'email': None,
              'institution': author[3] if len(author) > 3 else None,
            } for author in article.get('authors_parsed', [])
          ],
          'abstract': re.sub(r'\s+', ' ', article.get('abstract', '')).strip(),
          'keywords': None,
          'date': datetime.strptime(article['versions'][-1]['created'], '%a, %d %b %Y %H:%M:%S %Z').strftime('%Y-%m-%d')
        })

      except Exception as e:
        print(f"\n[{idx:06d}] Error while processing article \"{article_id}\": {e}")

        continue

  with open(output_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=2)

if __name__ == '__main__':
  main()
