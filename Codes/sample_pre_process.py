import csv
import os
import pandas as pd
import h5py

# Input file paths
file1 = "./dict.tsv"  # Format: ID \t SEQUENCE
file2 = "./pairs.tsv"          # Format: ID1 \t ID2 \t Label (0 or 1)

# Output file path
output_file = "sequence_pairs_output.tsv"

# Step 1: Load protein sequences from file1
id_to_sequence = {}
with open(file1, "r") as f1:
    reader = csv.reader(f1, delimiter="\t")
    for row in reader:
        if len(row) != 2:
            continue  # Skip malformed lines
        protein_id, sequence = row
        id_to_sequence[protein_id.strip()] = sequence.strip()

# Step 2: Read file2 and map IDs to sequences, write output with headers
with open(file2, "r") as f2, open(output_file, "w", newline="") as out:
    writer = csv.writer(out, delimiter="\t")
    
    # Write column headers
    writer.writerow(["protein_sequence_1", "protein_sequence_2", "label"])
    
    reader = csv.reader(f2, delimiter="\t")
    for row in reader:
        if len(row) != 3:
            continue  # Skip malformed lines
        id1, id2, label = row
        seq1 = id_to_sequence.get(id1.strip())
        seq2 = id_to_sequence.get(id2.strip())
        if seq1 and seq2:
            writer.writerow([seq1, seq2, label.strip()])
        else:
            print(f"⚠️ Missing sequence for ID: {id1} or {id2}")

# Configuration
input_tsv = "sequence_pairs_output.tsv"  # Update this if your file is named differently
output_dir = "dataset2_batches_h5"
batch_size = 50

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read the TSV file
df = pd.read_csv(input_tsv, sep="\t", header=None, names=["protein_sequence_1", "protein_sequence_2", "label"])

# Split into batches and save each to HDF5
for i in range(0, len(df), batch_size):
    batch_df = df.iloc[i:i+batch_size]
    batch_idx = i // batch_size
    output_path = os.path.join(output_dir, f"batch_{batch_idx:03}.h5")
    
    with h5py.File(output_path, "w") as h5f:
        h5f.create_dataset("protein_sequence_1", data=batch_df["protein_sequence_1"].astype("S").to_numpy())
        h5f.create_dataset("protein_sequence_2", data=batch_df["protein_sequence_2"].astype("S").to_numpy())
        h5f.create_dataset("label", data=batch_df["label"].to_numpy())
    
    print(f"Saved {output_path} with {len(batch_df)} rows")