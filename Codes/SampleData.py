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
import tqdm
import gc
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import requests
from io import StringIO
import re
import h5py as hf


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

output_file = "all_embeddings.h5"
data = {}

with h5py.File(output_file, "r") as f:
    for seq in f.keys():
        embedding = f[seq][()]  # Get the NumPy array
        data[seq] = embedding

def embed_pairs(sequence_pairs_pos, sequence_pairs_neg, embedding_dict):
    """Replace sequence with embedding. Remove pairs if embedding is missing."""
    embedded_data = []
    for seq1, seq2 in sequence_pairs_pos:
        label = 1
        if seq1 in embedding_dict and seq2 in embedding_dict:
            embedded_data.append(
                {
                    "sequence_a": embedding_dict[seq1].flatten(),
                    "sequence_b": embedding_dict[seq2].flatten(),
                    "label": 1
                }
            )
    for seq1, seq2 in sequence_pairs_neg:
        label = 0
        if seq1 in embedding_dict and seq2 in embedding_dict:
            embedded_data.append(
                {
                    "sequence_a": embedding_dict[seq1].flatten(),
                    "sequence_b": embedding_dict[seq2].flatten(),
                    "label": 0
                }
            )
    df_em = pd.DataFrame(embedded_data)
    return df_em

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import classification_report
model_rd = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model_rd.fit(X_train, y_train)


y_pred = model_rd.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

from xgboost import XGBClassifier
modelxg = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
modelxg.fit(X_train, y_train)

y_pred = modelxg.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred))
print(classification_report(y_test, y_pred))



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

model_lr = LogisticRegression(class_weight='balanced', max_iter=5000)
model_lr.fit(X_train, y_train)
y_pred = model_lr.predict(X_test)
print(classification_report(y_test, y_pred))