## NLP Stress Project

This repository includes scripts to reproduce the analyses reported in: Antonacci, C., Uy, J. P., Kwan, K., Giampetruzzi, E., Jones, S., Pennebaker, J. W., & Gotlib, I. H. (2025). **The language of stress: Leveraging natural language processing to model risk for psychopathology across adolescence.** Manuscript Submitted for Publication.

It contains the complete pipeline for processing and analyzing **TESI (Traumatic Events Screening Inventory) stress interviews** from the **Early Life Stress, Puberty, and Neural Trajectories (ELS) study** in the Stanford Neurodevelopment, Affect, and Psychopathology Laboratory (PI: Ian Gotlib). The workflow spans **raw audio transcription**, **speaker correction**, **NLP feature generation**, and **predictive modeling** of internalizing outcomes. 


---

### Workflow Overview

The pipeline consists of three main phases:

#### 1. Transcription and Parsing

These scripts convert raw audio files into cleaned, **participant-only transcripts**, which serve as the foundation for all NLP analyses.

* `whisperX_batch.sh`: SLURM job-array script for large-scale **WhisperX** transcription and diarization.
* `whisperX_process.py`: Runs WhisperX: audio conversion, `large-v2` transcription, alignment, and **PyAnnote** diarization.
* `GPT-4o_Standardized_Prompt.txt`: Standardized **GPT-4o prompt** used to correct diarization errors (ensures participant vs. interviewer turns are accurate).
* `Speaker_Parsing.py`: Extracts *only Participant speech* and produces the final transcript used across all NLP analyses.

---

#### 2. NLP Feature Generation

Multiple NLP representations are generated from participant-only transcripts.

##### TF-IDF Features

* `TF-IDF_Generation.py`: Produces a filtered document–term matrix ($\ge 5\%$ document frequency).

##### RoBERTa Embeddings

* `Generate_Embeddings.py` Outputs:
    * `roberta_embeddings.csv`: 768-dimensional pooled participant embeddings.
    * `sentence_risk_scores.csv`: Sentence-level projection scores for interpretability.

##### LDA Topic Modeling (DLATK + MALLET)

The topic modeling pipeline uses sentence-level **MALLET LDA**, participant-level aggregation, and PCA reduction to derive higher-order thematic components.

- `01_preprocess_transcripts.py`
  Sentence-level tokenization and creation of a **DLATK-ready corpus**.

- `02_lda_to_topics.py`
  Runs **MALLET LDA (~500 topics)**. Exports: topic–word distributions, sentence-level topic proportions, and participant-level aggregated topic distributions.  

---

#### 3. Statistical Modeling (R Markdown)

These notebooks reproduce all predictive models and figures in the manuscript.

* **`LIWC_Analyses.Rmd`**
    * **Features:** LIWC
    * **Analyses:** Baseline LIWC models, Elastic net prediction, PCA of LIWC dimensions.

* **`TF-IDF_Analyses.Rmd`**
    * **Features:** TF-IDF
    * **Analyses:** Elastic net prediction, PCA identifying the Function Words component.

* **`LDA_Analyses.Rmd`**
    * **Features:** LDA Topics (20 PCA-reduced components)
    * **Analyses:** Elastic net prediction Identification of five retained components, Wordcloud visualization.

* **`SentenceEmbedding_Analysis.Rmd`**
    * **Features:** RoBERTa Embeddings
    * **Analyses:** Elastic net, UMAP + HDBSCAN clustering, Projection of coefficients onto individual sentences, Interpretive analyses.
