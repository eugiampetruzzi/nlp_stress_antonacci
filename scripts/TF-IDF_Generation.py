"""
This script corresponds to the TF-IDF Methods section of the manuscript.

It performs the following steps:
1.  Loads all participant transcripts from a specified folder.
2.  Computes a document-term matrix of TF-IDF scores.
3.  Filters this matrix to remove words that appear in <5% of transcripts.
4.  Saves the final, filtered feature matrix (participants x features) to a CSV
    file, which is then used in the 'TF_IDF_analyses.Rmd' script.
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# --- 1. Setup and Constants ---

# define relative paths
# assumes data is in 'data/Transcripts'
data_folder = os.path.join("data", "Transcripts")
output_path = os.path.join("data", "tfidf_features_filtered.csv")

# regex for word tokens (2+ alphabetic characters)
TOKEN_PATTERN = r'(?u)\b[a-zA-Z][a-zA-Z]+\b'

# frequency threshold (must appear in at least 5% of documents)
MIN_DOC_FREQUENCY = 0.05


# --- 2. Load Transcripts ---
print(f"Loading transcripts from: {data_folder}")

transcripts = []
labels = []

try:
    audio_files = sorted(os.listdir(data_folder))
except FileNotFoundError:
    print(f"Error: Transcript folder not found at {data_folder}")
    print("Please check the 'data_folder' path.")
    # exit() # or raise error

for filename in audio_files:
    if filename.endswith(".txt"):
        with open(os.path.join(data_folder, filename), "r", encoding="utf-8") as file:
            transcripts.append(file.read())
        
        # extract subject ID (e.g., 'ELS_123.txt' -> 123)
        # note: this assumes a specific 'ELS_XXX.txt' format
        try:
            subject_id = int(filename[4:7])
            labels.append(subject_id)
        except ValueError:
            print(f"Warning: Could not parse ELS_ID from filename: {filename}")
        
    print(f"Loaded subject {subject_id}'s transcript.")

print(f"Finished loading all transcripts.")


# --- 3. Calculate TF-IDF ---
print("Calculating TF-IDF matrix...")

# initialize the vectorizer
# default settings correctly compute TF * IDF
vectorizer = TfidfVectorizer(token_pattern=TOKEN_PATTERN)

# fit and transform the data
tfidf_matrix = vectorizer.fit_transform(transcripts)
feature_names = vectorizer.get_feature_names_out()

# convert to a DataFrame
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

print(f"Initial matrix shape (subjects, features): {df_tfidf.shape}")

# --- 4. Filter Sparse Features (<5% Document Frequency) ---

# calculate the sparsity threshold
# (e.g., 0.95 means 0 in >95% of docs, same as appearing in <5%)
sparsity_threshold = 1.0 - MIN_DOC_FREQUENCY

# get a boolean mask for sparse features
sparse_mask = (df_tfidf == 0).sum(axis=0) > sparsity_threshold * len(df_tfidf)

# apply the mask to drop sparse columns
filtered_tfidf = df_tfidf.loc[:, ~sparse_mask]

# add participant labels as the first column
filtered_tfidf.insert(0, "ELS_ID", labels)


# --- 5. Save Filtered Data ---

filtered_tfidf.to_csv(output_path, index=False)

print(f"Filtered matrix shape (subjects, features): {filtered_tfidf.shape}")
print(f"  (This matches the {filtered_tfidf.shape[1]-1} features from METHODS)")
print(f"Filtered TF-IDF features saved to: {output_path}")