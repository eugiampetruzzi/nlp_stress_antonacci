"""
This script generates participant-level and sentence-level embeddings from
raw text transcripts using a RoBERTa model.

1.  Generates 768-dim participant-level embeddings by:
    a.  Splitting transcripts into sentences.
    b.  Encoding each sentence and normalizing its embedding to unit length.
    c.  Mean-pooling all normalized sentence vectors for a participant.
    d.  Re-normalizing the final mean-pooled vector.
    e.  Saves the result to 'roberta_embeddings.csv' (for the R model).

2.  Generates sentence-level "risk scores" by:
    a.  Loading the 768-dim coefficient vector from the R elastic net model
        (from 'roberta_embedding_coefficients.csv').
    b.  Calculating the dot product (via matrix multiplication '@')
        of the model vector and each normalized sentence embedding.
    c.  Saves the results (ELS_ID, Sentence, Risk_Score) to
        'sentence_risk_scores.csv' (for interpretation).
"""

import os
import re
import glob
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel

# --- 1. Setup ---
print("Initializing model and tokenizer...")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base")
model.eval()

# put model on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model moved to {device}.")

# define paths
base_path = os.path.expanduser(
    "~/Library/CloudStorage/Box-Box/Mooddata_Coordinating/ELS_RDoC/Interview_Recordings"
)
transcript_folder = os.path.join(base_path, "Participant_Parse")
output_folder = os.path.dirname(transcript_folder)


# --- 2. Helper Functions ---

def split_sentences(text: str):
    """Splits text into a list of sentences."""
    return [s.strip() for s in re.split(r'(?<=[.?!])\s+', text) if s.strip()]


def encode_sentence(sentence: str) -> torch.Tensor:
    """Return a 1D RoBERTa sentence embedding (mean-pooled, attention-masked)."""
    inputs = tokenizer(
        sentence, return_tensors="pt", truncation=True, max_length=512
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    last_hidden = outputs.last_hidden_state
    attn = inputs["attention_mask"].unsqueeze(-1)
    summed = (last_hidden * attn).sum(dim=1)
    counts = attn.sum(dim=1).clamp(min=1e-9)
    sent_vec = (summed / counts).squeeze(0)
    return sent_vec


def get_els_id(filename: str) -> str:
    """Extracts ELS ID (e.g., 'ELS_123') from a filename."""
    match = re.search(r"(ELS[_-]?\d+)", filename, flags=re.IGNORECASE)
    return match.group(1) if match else None


# --- 3. Main Processing Loop (Participant & Sentence Embeddings) ---

participant_embeddings = {}      # stores {els_id: final_participant_vector}
sentence_embeddings_by_file = {} # stores {els_id: torch.Tensor[num_sents, 768]}
sentence_text_by_file = {}       # stores {els_id: {sentence_idx: "text"}}

transcript_files = glob.glob(os.path.join(transcript_folder, "*.txt"))
print(f"Found {len(transcript_files)} transcript files. Processing...")

total_sentences = 0
for file_path in transcript_files:
    filename = os.path.basename(file_path)
    els_id = get_els_id(filename)

    if not els_id:
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        continue

    sentences = split_sentences(text)
    if not sentences:
        continue

    # stores for this participant only
    normalized_sentence_vecs = []
    sentence_text_map = {}

    for idx, s in enumerate(sentences):
        sent_vec = encode_sentence(s)
        norm_sent_vec = F.normalize(sent_vec, p=2, dim=0)
        
        normalized_sentence_vecs.append(norm_sent_vec)
        sentence_text_map[idx] = s

    total_sentences += len(sentences)

    # stack all sentence tensors for this participant
    sentence_tensor = torch.stack(normalized_sentence_vecs, dim=0)

    # store data for dot product analysis
    sentence_embeddings_by_file[els_id] = sentence_tensor
    sentence_text_by_file[els_id] = sentence_text_map

    # average the normalized sentence vectors (on CPU)
    transcript_vec = sentence_tensor.cpu().mean(dim=0)

    # re-normalize the final averaged vector to unit length
    final_transcript_vec = F.normalize(transcript_vec, p=2, dim=0)

    # store the final participant vector
    participant_embeddings[els_id] = final_transcript_vec.numpy()

    # Progress update
    print(f"Processed {els_id}: {len(sentences)} sentences.")

print(f"Processed {len(participant_embeddings)} transcripts.")
print(f"Found {total_sentences} total sentences.")


# --- 4. Save Participant-Level Embeddings (for R model) ---

embedding_df = pd.DataFrame.from_dict(participant_embeddings, orient='index')
embedding_df.columns = [f'emb_{i}' for i in range(embedding_df.shape[1])]
embedding_df = embedding_df.reset_index().rename(columns={'index': 'ELS_ID'})

output_path_participants = os.path.join(output_folder, "roberta_embeddings.csv")
embedding_df.to_csv(output_path_participants, index=False)
print(f"Participant embeddings saved to: {output_path_participants}")


# --- 5. Dot Product Projection (Sentence-Level Interpretation) ---

print("\nStarting dot product projection...")
try:
    # load the 768 coefficients saved from R
    coef_path = os.path.join(
        output_folder, "roberta_embedding_coefficients.csv"
    )
    coef_df = pd.read_csv(coef_path, index_col=0)
    
    # get the column name R used (it's 'x' or 's1' from as.matrix)
    coef_col_name = coef_df.columns[0]
    
    # create a dictionary for fast lookup
    coef_dict = coef_df[coef_col_name].to_dict()

    # build the 768-dim coefficient vector, ensuring correct emb_0...emb_767 order
    # explicitly cast to float32 to ensure compatibility
    coef_vector = np.array(
        [coef_dict[f'emb_{i}'] for i in range(768)], dtype=np.float32
    )
    
    print(f"Loaded {len(coef_vector)} coefficients successfully.")

    projection_rows = []
    
    # iterate through the dictionaries created in step 3
    for els_id, sent_tensor in sentence_embeddings_by_file.items():
        
        # convert tensor to numpy matrix for @
        # [num_sentences, 768]
        # use .detach() to explicitly remove from computation graph
        M = sent_tensor.detach().cpu().numpy().astype(np.float32)
        
        # perform matrix-vector multiplication
        # [num_sents, 768] @ [768] -> [num_sents]
        scores = M @ coef_vector
        
        # get the corresponding text map
        text_map = sentence_text_by_file[els_id]
        
        for idx, score in enumerate(scores):
            projection_rows.append({
                "ELS_ID": els_id,
                "Sentence": text_map.get(idx, ""), # find text by index
                "Sentence_Index": idx,
                "Risk_Score": float(score)
            })

    # create and save the final DataFrame
    projection_df = pd.DataFrame(projection_rows)
    projection_df = projection_df.sort_values(by="Risk_Score", ascending=False)

    output_path_sentences = os.path.join(
        output_folder, "sentence_risk_scores.csv"
    )
    projection_df.to_csv(
        output_path_sentences, index=False, encoding='utf-8'
    )
    
    print(f"Sentence-level risk scores saved to: {output_path_sentences}")
    print("\nDot product projection complete.")

    # --- Robustness Fix ---
    # wrap the final print in a new try/except, as this line
    # can fail in some environments due to numpy/pandas repr issues
    try:
        print("Top 5 most 'at-risk' sentences:")
        print(projection_df.head(5))
    except Exception as print_e:
        print(f"\nCould not print .head() due to environment error: {print_e}")
        print("However, the 'sentence_risk_scores.csv' file was saved successfully.")

except FileNotFoundError:
    print(f"\nCRITICAL ERROR: Could not find coefficient file at {coef_path}")
    print("Please ensure you have run the R script 'Sentence_Embedding_analyses.Rmd'")
    print("and that 'roberta_embedding_coefficients.csv' exists in the data folder.")
except Exception as e:
    print(f"\nAn error occurred during the dot product projection: {e}")