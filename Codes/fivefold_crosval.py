import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from joblib import Parallel, delayed
import subprocess
import kagglehub
from tqdm import tqdm
import gc
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import requests
from io import StringIO
import re


def load_motif_data(url= "http://elm.eu.org/elms/elms_index.tsv"):
  url = "http://elm.eu.org/elms/elms_index.tsv"
  response = requests.get(url)

  # Check if the request was successful
  if response.status_code == 200:
      data = StringIO(response.text)
      df = pd.read_csv(data, sep='\t', skiprows=5)
      return df
  else:
      print("Failed to fetch data. Status code:", response.status_code)
      return None


elm_motifs = {
    row["ELMIdentifier"]: re.compile(row["Regex"])
    for _, row in df.iterrows()
}

def encode_motifs(sequence):
    """Convert a protein sequence to a binary motif vector."""
    return [1 if pattern.search(sequence) else 0 for pattern in elm_motifs.values()]

df = load_motif_data()

def augumented_data(pos_, neg_):
  data = []
  for seq in pos_:
    # Encode motifs for each protein
    seq1, seq2 = seq[0], seq[1]
    label = 1
    motifs_p1 = encode_motifs(seq1)
    motifs_p2 = encode_motifs(seq2)
    # Combine features and label
    data.append(motifs_p1 + motifs_p2 + [label])
  for seq in neg_:
    # Encode motifs for each protein
    seq1, seq2 = seq[0], seq[1]
    label = 0
    motifs_p1 = encode_motifs(seq1)
    motifs_p2 = encode_motifs(seq2)
    # Combine features and label
    data.append(motifs_p1 + motifs_p2 + [label])
  return data

def load_protein_data(url="spandansureja/ppi-dataset", sample_frac=1.0, random_state=42):
  dataset_path = kagglehub.dataset_download(url)
  pos_df = pd.read_csv(
        os.path.join(dataset_path, "positive_protein_sequences.csv"),
        nrows=int(36652*sample_frac)
    )
  neg_df = pd.read_csv(
        os.path.join(dataset_path, "negative_protein_sequences.csv"),
        nrows=int(36480*sample_frac)
    )
  # Convert to list of tuples
  positive_pairs = list(zip(pos_df['protein_sequences_1'], pos_df['protein_sequences_2']))
  negative_pairs = list(zip(neg_df['protein_sequences_1'], neg_df['protein_sequences_2']))

  return positive_pairs, negative_pairs


def load_data_from_file(file_name):
    # Read TSV with tab separator and strip any whitespace in headers
    df = pd.read_csv(file_name, sep="\t")
    df.columns = df.columns.str.strip()  # Remove extra whitespace from column names
    # Extract data as list of tuples
    data = []
    # data_pairs = list(zip(df["protein_sequence_1"], df["protein_sequence_2"], df["label"]))
    for _, row in df.iterrows():
        seq1 = row["protein_sequence_1"]
        # print(seq1)
        seq2 = row["protein_sequence_2"]
        label = row["label"]
        # print(f"{seq1[:10]}\t {seq2[:10]} \t {label} ")
        motifs_p1 = encode_motifs(seq1)
        # print(len(motifs_p1))
        motifs_p2 = encode_motifs(seq2)
        # print(len(motifs_p2))
        data.append(motifs_p1 + motifs_p2 + [label])
    return data