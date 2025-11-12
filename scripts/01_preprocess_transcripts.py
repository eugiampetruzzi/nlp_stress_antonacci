"""
01_preprocess_transcripts.py

prepares participant-only tesi stress interview transcripts for sentence-level lda with dlatk/mallet.
this script tokenizes each participant transcript into sentences and writes a dlatk-ready csv.

input:
  - a directory of participant-only .txt files (one file per participant)

output:
  - data/sentences_for_dlatk.csv with columns:
      id        (participant id from filename)
      sentence  (single sentence of text)
"""

import argparse
from pathlib import Path

import nltk
import pandas as pd

# download punkt tokenizer if not already present
nltk.download("punkt", quiet=True)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="tokenize participant transcripts into sentences for dlatk/mallet lda"
  )
  parser.add_argument(
    "--input_dir",
    type=str,
    default="data/participant_transcripts",
    help="directory with participant-only .txt transcripts (one file per participant)",
  )
  parser.add_argument(
    "--output_csv",
    type=str,
    default="data/sentences_for_dlatk.csv",
    help="output csv file with columns: id, sentence",
  )
  return parser.parse_args()


def load_transcript(path: Path) -> str:
  """read a single transcript text file"""
  with path.open("r", encoding="utf-8") as f:
    text = f.read()
  return text


def transcripts_to_sentences(input_dir: Path) -> pd.DataFrame:
  """
  iterate over .txt files, tokenize into sentences,
  and construct a dataframe with one row per sentence and participant id
  """
  rows = []

  txt_files = sorted(input_dir.glob("*.txt"))
  for fp in txt_files:
    participant_id = fp.stem
    text = load_transcript(fp)

    sentences = nltk.sent_tokenize(text)

    for sent in sentences:
      sent_clean = sent.strip()
      if not sent_clean:
        continue
      rows.append(
        {
          "id": participant_id,
          "sentence": sent_clean,
        }
      )

  return pd.DataFrame(rows)


def main() -> None:
  args = parse_args()

  input_dir = Path(args.input_dir)
  output_csv = Path(args.output_csv)
  output_csv.parent.mkdir(parents=True, exist_ok=True)

  sentences_df = transcripts_to_sentences(input_dir=input_dir)
  sentences_df.to_csv(output_csv, index=False)


if __name__ == "__main__":
  main()
