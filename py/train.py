import numpy as np
import pandas as pd
import networkx as nx
import os
import json
import torch

# %% md
# Load data
# %%
bp_db = pd.read_csv("../data/GOannotations_kept.csv", index_col=0)
bp_db.head()
# %%
counts1 = pd.read_csv("../data/counts1.csv", index_col=0)
counts1.head()
# %%
pheno1 = pd.read_csv("../data/pheno1.csv", index_col=0)
pheno1.drop(["diagnosis"], axis=1, inplace=True)

pheno1.head()
# %% md
# process the data
# %%
# add age, sex, lithium of pheno1 to counts1
tmp_pheno1 = pheno1[["age", "sex", "lithium"]].apply(lambda x: x.replace("M", 0).replace("F", 1))  # chagne sex to 0, 1
counts1_merge = pd.merge(counts1, tmp_pheno1, left_index=True, right_index=True)

counts1_merge = (counts1_merge - counts1_merge.mean()) / counts1_merge.std()
counts1_merge.head()
# %%
mask_features = counts1.columns.isin(bp_db["ENSEMBL"])
mask_features
# %%
gnn_dataset = {}
gnn_dataset['x'] = counts1.to_numpy()
gnn_dataset['y'] = pheno1['condition'].to_numpy()
# %%
bp_db_genes = set(bp_db["ENSEMBL"])
list_genes = [x for x in counts1.columns if x not in bp_db_genes]
gp_go = bp_db.groupby("GO")
list_go = list(gp_go.groups.keys())

index_genes = {gene: idx for idx, gene in enumerate(list_genes)}
index_go = {go: idx for idx, go in enumerate(list_go)}

matrix_connection = torch.tensor(np.zeros((len(list_genes), len(list_go)), dtype=np.float32))
for idx, row in bp_db.iterrows():
    gene = row["ENSEMBL"]
    go = row["GO"]
    if gene in index_genes and go in index_go:
        matrix_connection[index_genes[gene], index_go[go]] = 1

a = pd.DataFrame(matrix_connection)

graph = nx.read_gml("../data/bp_graph.gml")
print(graph)

df_go_level = pd.read_csv("../data/go_to_level.csv", index_col=0)
df_go_level.head()

with open("../data/map_int_go.txt", 'r') as fp:
    map_int_go = json.load(fp)
map_int_go = {int(idx): go for idx, go in map_int_go.items()}
[x for x in map_int_go.items()][:5]

map_go_int = {go: idx for idx, go in map_int_go.items()}
[x for x in map_go_int.items()][:5]

print()
device = torch.device("cuda:2")
import torch
import numpy as np
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
np.random.seed(12345)
edge_index = np.random.randint(0, 444, (2, 3000), dtype=np.int64)  # Simulating random connections

x = torch.tensor(gnn_dataset['x'], dtype=torch.float32).to(device)
y = torch.tensor((gnn_dataset['y'] != 'Control').astype(int), dtype=torch.long).to(device)
edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)

data = Data(x=x, edge_index=edge_index, y=y)

data = data.to(device)
print()
data.x = (data.x - data.x.mean()) / data.x.std()


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

model = GCN(num_node_features=data.num_node_features, num_classes=2)
model = model.to(device)
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)
        # Maybe add or remove some layers, or adjust the number of units

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)  # Adjust dropout rate
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

model = GCN(num_node_features=data.num_node_features, num_classes=2)
model = model.to(device)
from torch.optim import Adam
from sklearn.model_selection import train_test_split

# Split data into training and test indices
train_idx, test_idx = train_test_split(range(444), test_size=0.1, random_state=42)
train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
train_mask[train_idx] = True
train_mask = train_mask.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[train_mask], data.y[train_mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()
for epoch in range(200):
    loss = train()
    print(f'Epoch {epoch+1}: Loss: {loss:.4f}')
def test():
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    correct = (pred[test_idx] == data.y[test_idx]).sum()
    acc = correct.item() / len(test_idx)
    return acc

accuracy = test()
print(f'Test Accuracy: {accuracy:.4f}')
print()