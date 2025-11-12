"""
02_lda_to_topics.py

aggregates sentence-level lda topic proportions to participant-level topic distributions.

this script corresponds to the lda feature construction:
  - sentence-level mallet lda (~500 topics) using dlatk
  - averaging sentence-level topic proportions within participant to get a
    participant-topic distribution for downstream pca and modeling

expected input:
  - csv with columns: id, topic_1, topic_2, ..., topic_k

output:
  - data/participant_topic_distribution.csv
"""

import argparse
from pathlib import Path
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="aggregate sentence-level lda topic proportions to participant-level means"
    )
    parser.add_argument(
        "--sentence_topics_csv",
        type=str,
        default="data/sentence_topic_proportions.csv",
        help="csv with sentence-level topic proportions (one row per sentence)",
    )
    parser.add_argument(
        "--participant_topics_csv",
        type=str,
        default="data/participant_topic_distribution.csv",
        help="output csv with participant-level mean topic proportions",
    )
    return parser.parse_args()


def aggregate_sentence_topics(sentence_topics_csv: Path,
                              participant_topics_csv: Path) -> None:
    """
    aggregate sentence-level topics to participant-level averages

    input columns:
      - id
      - topic_1, topic_2, ..., topic_k

    output columns:
      - id
      - topic_1, topic_2, ..., topic_k (mean across sentences)
    """
    df = pd.read_csv(sentence_topics_csv)

    if "id" not in df.columns:
        raise ValueError("input file must contain an 'id' column")

    topic_cols = [c for c in df.columns if c.startswith("topic_")]
    if not topic_cols:
        raise ValueError("no topic_ columns found")

    grouped = (
        df.groupby("id", as_index=False)[topic_cols]
          .mean()
    )

    participant_topics_csv.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(participant_topics_csv, index=False)


def main() -> None:
    args = parse_args()

    sentence_topics_csv = Path(args.sentence_topics_csv)
    participant_topics_csv = Path(args.participant_topics_csv)

    aggregate_sentence_topics(
        sentence_topics_csv=sentence_topics_csv,
        participant_topics_csv=participant_topics_csv,
    )


if __name__ == "__main__":
    main()
