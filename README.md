# nlp_stress_antonacci

This repository contains the full pipeline for processing and analyzing TESI (Traumatic Events Screening Inventory) stress interviews from the Early Life Stress, Puberty, and Neural Trajectories (ELS) study in the Stanford Neurodevelopment, Affect, and Psychopathology Laboratory (PI: Ian Gotlib). The workflow spans raw audio transcription, diarization correction, NLP feature generation, and predictive modeling of later internalizing outcomes.

All scripts reproduce the analyses in the manuscript.

Workflow Overview

The repository is organized into three main phases:

⸻

1. Transcription and Parsing

These scripts convert raw audio files into cleaned, participant-only text transcripts, which serve as the input for every NLP method.

Scripts
	•	whisperX_batch.sh
SLURM job-array script for large-scale WhisperX transcription + diarization.
	•	whisperX_process.py
Runs WhisperX end-to-end: audio conversion, large-v2 transcription, timestamp alignment, and PyAnnote diarization.
	•	GPT-4o_Standardized_Prompt.txt
The standardized GPT-4o prompt used to correct diarization errors (ensuring participant vs. interviewer turns are accurate).
	•	Speaker_Parsing.py
Extracts only Participant speech from GPT-4o-corrected transcripts and outputs one cleaned transcript per participant.

These participant-only transcripts are the foundation for all NLP feature extraction.

⸻

2. NLP Feature Generation

Multiple NLP representations are generated from the cleaned participant text.

TF-IDF
	•	TF-IDF_Generation.py
Creates a filtered TF-IDF document–term matrix (≥5% document frequency).

RoBERTa Embeddings
	•	Generate_Embeddings.py
Produces:
	•	roberta_embeddings.csv: 768-dimensional mean-pooled embeddings
	•	sentence_risk_scores.csv: sentence-level projection scores for interpretation

LDA Topic Modeling (DLATK + MALLET)

The updated LDA pipeline uses sentence-level LDA, participant-level aggregation, and PCA for dimensionality reduction.

Scripts:
	•	01_preprocess_transcripts.py — sentence tokenization + DLATK-ready corpus
	•	02_lda_to_topics.py — runs MALLET LDA (~500 topics), exports:
	•	topic–word distributions
	•	sentence-level topic proportions
	•	participant-level topic distributions

These outputs feed directly into LDA_Analyses.Rmd.

⸻

3. Statistical Modeling (R Markdown)

These notebooks reproduce all analyses and figures reported in the manuscript.

LIWC_Analyses.Rmd
	•	Baseline LIWC models
	•	LIWC elastic net prediction
	•	PCA of LIWC features

TF-IDF_Analyses.Rmd
	•	Elastic net prediction from TF-IDF
	•	PCA identifying the “Function Words” dimension

LDA_Analyses.Rmd

Uses the 20 PCA-reduced LDA topic components.

Includes:
	•	Elastic net prediction of YSR internalizing at T3
	•	Logistic regression for incident diagnosis
	•	Retained component identification
	•	Wordcloud interpretation (risk vs. protective themes)

SentenceEmbedding_Analysis.Rmd
	•	Elastic net on RoBERTa embeddings
	•	UMAP/HDBSCAN clustering
	•	Saving the coefficient vector for sentence-level projections
	•	Predictive modeling and cluster interpretation
