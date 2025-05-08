# scripts/generate_training_data.py

import pandas as pd
import sys

# Read command-line arguments
input_file = sys.argv[1]    # training input (gzipped TSV)
output_file = sys.argv[2]   # path to save processed training data

# Step 1: Load and Process Training Data
training_data = pd.read_csv(input_file, sep="\t", on_bad_lines="skip")
training_data.rename(columns={"NCBI.GeneID.Target": "node1_id", "NCBI.GeneID.TF": "node2_id"}, inplace=True)
training_data.fillna("", inplace=True)

#  Prepend 'NCBIGene:' to node1_id and node2_id
training_data["node1_id"] = "NCBIGene:" + training_data["node1_id"].astype(str)
training_data["node2_id"] = "NCBIGene:" + training_data["node2_id"].astype(str)

# Step 2: Create all possible negative pairs
all_node1 = training_data["node1_id"].unique()
all_node2 = training_data["node2_id"].unique()

# Cartesian product
all_pairs = pd.MultiIndex.from_product([all_node1, all_node2], names=["node1_id", "node2_id"]).to_frame(index=False)

# Mark whether each pair exists
existing_pairs = set(zip(training_data["node1_id"], training_data["node2_id"]))
all_pairs["interaction"] = all_pairs.apply(lambda row: (row["node1_id"], row["node2_id"]) in existing_pairs, axis=1)

# Step 3: Save the final training data
all_pairs.to_csv(output_file, sep="\t", index=False)
