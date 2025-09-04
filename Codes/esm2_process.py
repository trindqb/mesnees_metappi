import h5py
# Folder containing batches of sequences
input_folder = "sequence_batches_h5"
output_file = "all_embeddings.h5"
with hf.File(output_file, "w") as f:
    pass
# Collect all input files
batch_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".h5")])
for batch_file in tqdm(batch_files, desc="Process file"):
    batch_path = os.path.join(input_folder, batch_file)
    with h5py.File(batch_path, "r") as f:
        sequences = f["sequences"][:]
        sequences = [s.decode("utf-8") if isinstance(s, bytes) else s for s in sequences]
    print(len(sequences))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # device = torch.device("cpu")
    
    # Load ESM-2 8M model and move to device
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    for seq in tqdm(sequences, desc="Encoding all sequences"):
        try:
            batch_data = [("protein1", seq)]
            labels, strs, tokens = batch_converter(batch_data)
            tokens = tokens.to(device)
            with torch.no_grad():
                results = model(tokens, repr_layers=[6], return_contacts=False)
            embedded = results["representations"][6]
            protein_embedding = embedded.mean(dim=1).cpu().numpy()  # shape: (1, 320)
            # Save to the single output file
            with hf.File(output_file, "a") as f_out:
                f_out.create_dataset(
                    name=f"{seq}",
                    data=protein_embedding
                )
                # Cleanup
            del tokens, results, embedded, protein_embedding
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"⚠️ Error processing sequence #: {e}")