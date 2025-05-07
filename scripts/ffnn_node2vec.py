import os
import sys
import gzip
import json
import pickle
import torch
import numpy as np
import pandas as pd
import networkx as nx
from nodevectors import Node2Vec
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Turn off tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Arguments
training_data_path = sys.argv[1]
kg2_nodes_path = sys.argv[2]
kg2_edges_path = sys.argv[3]

# Set paths to save outputs
output_dir = sys.argv[4]  # Pass an output directory in Snakemake
os.makedirs(output_dir, exist_ok=True)

# Loaders
def load_jsonl(file_path):
    if file_path.endswith('.gz'):
        open_func = gzip.open
        mode = 'rt'  # Read text mode
    else:
        open_func = open
        mode = 'r'
    
    with open_func(file_path, mode) as f:
        return [json.loads(line) for line in f]

# 1. Load data
data = pd.read_csv(training_data_path, sep='\t', header=None, names=['node2_id', 'node1_id', 'interaction'], skiprows=1)

# Balance dataset
min_count = 100000
balanced_data = pd.concat([
    data[data['interaction'] == False].sample(n=min_count, random_state=42),
    data[data['interaction'] == True].sample(n=min_count, random_state=42)
]).sample(frac=1, random_state=42)

# 2. Build Graph
nodes = load_jsonl(kg2_nodes_path)
edges = load_jsonl(kg2_edges_path)

G = nx.Graph()
for node in nodes:
    G.add_node(node['id'], name=node.get('name', ""), description=node.get('description', ""), category=node.get('category', ""))

for edge in edges:
    G.add_edge(edge['subject'], edge['object'])

# Save graph
with open(os.path.join(output_dir, 'graph.pkl'), 'wb') as f:
    pickle.dump(G, f)

# 3. Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4. Load BERT
tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
bert_model = AutoModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext').to(device)

def embed_description(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    outputs = bert_model(**inputs)
    return torch.mean(outputs.last_hidden_state, dim=1).detach().cpu().numpy().flatten()

# 5. Node2Vec Embeddings
n2v_path = os.path.join(output_dir, 'node2vec')

if os.path.exists(n2v_path):
    n2v = Node2Vec.load(n2v_path)
else:
    n2v = Node2Vec(n_components=64, walklen=10, epochs=50, return_weight=1.0, neighbor_weight=1.0, threads=16, w2vparams={'window': 100})
    n2v.fit(G)
    n2v.save(n2v_path)

# Save node2vec embeddings separately
node_ids = []
node_embeddings = []

for node in G.nodes():
    try:
        emb = n2v.predict(str(node))
        node_ids.append(node)
        node_embeddings.append(emb)
    except KeyError:
        print(f"Node {node} missing from Node2Vec model, skipping.")

np.save(os.path.join(output_dir, 'node_ids.npy'), np.array(node_ids))
np.save(os.path.join(output_dir, 'node2vec_embeddings.npy'), np.array(node_embeddings))

# 6. Prepare features
features = []
labels = balanced_data['interaction'].values

print("Preparing Features...")
for _, row in balanced_data.iterrows():
    try:
        node1_emb = n2v.predict(str(row['node1_id']))
        node2_emb = n2v.predict(str(row['node2_id']))
        node1_desc_emb = embed_description(G.nodes[row['node1_id']].get('description', ""))
        node2_desc_emb = embed_description(G.nodes[row['node2_id']].get('description', ""))
        feature_vector = np.concatenate((node1_emb, node2_emb, node1_desc_emb, node2_desc_emb))
        features.append(feature_vector)
    except KeyError as e:
        print(f"Skipping node pair {row['node1_id']} - {row['node2_id']} due to error: {e}")

features = np.array(features)
labels = np.array(labels, dtype=np.float32)

# 7. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 8. Model definition
class FeedforwardNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

input_size = X_train.shape[1]
hidden_size = 64

model = FeedforwardNN(input_size, hidden_size).to(device)

# 9. Train model
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 300
batch_size = 32

for epoch in range(num_epochs):
    model.train()
    permutation = torch.randperm(X_train.shape[0])

    for i in range(0, X_train.shape[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_X = torch.tensor(X_train[indices], dtype=torch.float32).to(device)
        batch_y = torch.tensor(y_train[indices], dtype=torch.float32).unsqueeze(1).to(device)

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 10. Evaluate
model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

    outputs = model(X_test_tensor)
    predictions = (outputs >= 0.5).float()
    accuracy = accuracy_score(y_test_tensor.cpu(), predictions.cpu())

print(f"Test Accuracy: {accuracy:.4f}")

# 11. Save model
torch.save(model.state_dict(), os.path.join(output_dir, 'ffnn_node2vec.pth'))

print(f"Training complete. Artifacts saved to {output_dir}")
