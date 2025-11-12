# nlp_stress_antonacci

This repository contains all scripts necessary to reproduce the data processing and statistical analyses. The workflow is divided into three main phases: (1) Transcription and Parsing, (2) NLP Feature Generation, and (3) Statistical Modeling.
1. Transcription and Parsing Scripts
These scripts convert raw audio files into cleaned, participant-only text files.
whisperX_batch.sh: A SLURM batch script for running transcription and diarization on a cluster. It is configured as a job array to efficiently process a large number of audio files in parallel by submitting one job per file. This script handles setting up the environment and calling whisperX_process.py for each audio file.
whisperX_process.py: The core Python processing script called by whisperX_batch.sh. It takes a single audio file as input, converts it from .wma to .wav (if necessary), and runs the full WhisperX pipeline: large-v2 transcription, timestamp alignment, and PyAnnote speaker diarization. It outputs the raw, diarized transcript (e.g., with SPEAKER_00 labels).
GPT-4o_Standardized_Prompt.txt: A text file containing the exact, standardized prompt used in a secure GPT-4o deployment. This prompt instructs the model to correct speaker diarization errors (e.g., participant speech assigned to interviewer or vice versa) in the WhisperX transcripts.
Speaker_Parsing.py: A Python script that parses the GPT-4o-corrected transcripts. It iterates through each file, extracts only the lines labeled Participant, and concatenates them into a single text file per participant. These participant-only files are the final input for all NLP feature generation scripts.
2. NLP Feature Generation Scripts
These scripts take the participant-only text files and generate the feature matrices used in the models.
TF-IDF_Generation.py: A Python script that generates the TF-IDF features. It loads the participant-only transcripts, computes a document-term matrix of TF-IDF scores, and filters it to retain only features present in at least 5% of all documents.
Generate_Embeddings.py: A Python script that uses a pre-trained RoBERTa model to process participant-only transcripts. It generates two main outputs:
roberta_embeddings.csv: Mean-pooled 768-dimensional vectors per participant (for the R model).
sentence_risk_scores.csv: The result of the dot-product projection of the R model's coefficients onto each individual sentence (for interpretation).
3. Statistical Modeling Scripts (R Markdown)
These R Markdown files contain all the statistical analyses and figure generation reported in the manuscript.
LIWC_Analyses.Rmd: An R Markdown script for all LIWC-based analyses. This script contains the code for the baseline models, the primary LIWC elastic net model, and the subsequent PCA on LIWC features to test its predictive utility.
TF-IDF_Analyses.Rmd: An R Markdown script for all TF-IDF-based analyses. It loads the filtered TF-IDF features, runs the elastic net model, and performs the PCA to identify the "Function Words" dimension and test its predictive utility.
LDA_Analyses.Rmd: An R Markdown script that loads the 20 PCA-reduced topic dimensions (generated from LDA). It runs the elastic net regression and subsequent linear/logistic models to predict internalizing symptoms and diagnoses from these latent topics.
SentenceEmbedding_Analysis.Rmd: An R Markdown script for all RoBERTa embedding-based analyses. It runs the elastic net model on the 768-dim participant vectors, saves the resulting coefficient vector for the Python dot-product projection, and performs the UMAP/HDBSCAN clustering and subsequent predictive models.
