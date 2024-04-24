import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.nn import Node2Vec
from torch_geometric.data import Data
print(torch.__version__)
bp_db = pd.read_csv("../data/bp_db.csv", index_col=0)
counts1 = pd.read_csv("../data/counts1.csv", index_col=0)
pheno1 = pd.read_csv("../data/pheno1.csv", index_col=0)
pheno1.drop(["diagnosis"], axis=1, inplace=True)
pheno1["condition"] = pheno1["condition"].apply(lambda x: 0 if x == "Control" else 1)
bp_graph = nx.read_gml("../data/bp_graph.gml")
bp_db_go = sorted(set(bp_graph.nodes))

map_int_go = {int(idx): go for idx, go in enumerate(bp_db_go)}
map_go_int = {go: idx for idx, go in map_int_go.items()}

_graph = nx.relabel_nodes(bp_graph, map_go_int, copy=False)
edge_index = torch.tensor(list(_graph.edges()), dtype=torch.long).t().contiguous()

data = Data(x=None, edge_index=edge_index)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Node2Vec(data.edge_index, embedding_dim=8, walk_length=20,
                 context_size=10, walks_per_node=10,
                 num_negative_samples=1, p=1, q=1, sparse=True).to(device)

loader = model.loader(batch_size=4, shuffle=True)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.001)
def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)
for epoch in range(1, 10):
    loss = train()
    #acc = test()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
go_embedding = model()  # go_index -> go_embedding
_embedding = go_embedding.cpu().detach().numpy()

map_int_gene = {int(idx): gene for idx, gene in enumerate(counts1.columns)}
map_gene_int = {gene: idx for idx, gene in map_int_gene.items()}

# convert to gene_index -> go_embedding
embedding_gene = np.zeros((len(counts1.columns), _embedding.shape[1]))
for idx, gene in map_int_gene.items():
    if gene in map_go_int:
        embedding_gene[idx] = _embedding[map_go_int[gene]]

embedding_gene = torch.tensor(embedding_gene, dtype=torch.float32, device=device)
tmp_pheno1 = pheno1[["age", "sex", "lithium"]].apply(lambda x: x.replace("M", 0).replace("F", 1))  # chagne sex to 0, 1
counts1_merge = pd.merge(counts1, tmp_pheno1, left_index=True, right_index=True)

counts1_merge = (counts1_merge - counts1_merge.mean()) / counts1_merge.std()
bp_db_genes = set(bp_db.ENSEMBL)
dataset = []
input_dim = None

for row in counts1_merge.iterrows():
    idx = row[0]
    values = row[1]
    _data = {
        "gene_with_go_idx": [],
        "gene_with_go_value": [],
        "gene_without_go_value": [],
        "other_info": []
    }
    for k, v in values.items():
        if k in bp_db_genes:
            _data["gene_with_go_idx"].append(map_gene_int[k])
            _data["gene_with_go_value"].append(v)
        elif k in map_gene_int:
            _data["gene_without_go_value"].append(v)
        else:
            _data["other_info"].append(v)
    dataset.append(_data)
    if input_dim is None:
        input_dim = len(_data["gene_with_go_idx"]) * 2 + len(_data["gene_without_go_value"]) + len(_data["other_info"])
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = self.X[idx]
        gene_with_go_idx = torch.tensor(sample['gene_with_go_idx'], dtype=torch.long, device=device)
        gene_with_go_value = torch.tensor(sample['gene_with_go_value'], dtype=torch.float, device=device)
        gene_without_go_value = torch.tensor(sample['gene_without_go_value'], dtype=torch.float, device=device)
        other_info = torch.tensor(sample['other_info'], dtype=torch.float, device=device)

        return {
            'gene_with_go_idx': gene_with_go_idx,
            'gene_with_go_value': gene_with_go_value,
            'gene_without_go_value': gene_without_go_value,
            'other_info': other_info
        }, torch.tensor(self.y[idx], dtype=torch.long, device=device)
from sklearn.model_selection import train_test_split

X = dataset
y = pheno1["condition"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(len(X_train), len(X_test))
train_dataset = MyDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

test_dataset = MyDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self, node_embedding, input_dim, output_dim):
        super().__init__()

        self.embedding = node_embedding
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, X):
        gene_with_go_idx = X['gene_with_go_idx']  # (B, N_1)
        gene_with_go_value = X['gene_with_go_value']  # (B, N_1)
        gene_without_go_value = X['gene_without_go_value']  # (B, N_2)
        other_info = X['other_info']  # (B, N_3)

        gene_with_go_embedding = self.embedding[gene_with_go_idx]  # (B, N_1, 8)
        gene_with_go_embedding = gene_with_go_embedding * gene_with_go_value.unsqueeze(-1)  # (B, N_1, 8)
        gene_with_go_embedding = gene_with_go_embedding.mean(dim=2)  # (B, N_1)

        output = torch.cat([gene_with_go_embedding, gene_with_go_value, gene_without_go_value, other_info],
                           dim=1)  # (B, N_1 * 2 + N_2 + N_3)

        output = self.fc1(output)  # (B, 256)
        output = self.relu1(output)
        output = self.fc2(output)  # (B, 128)
        output = self.relu2(output)
        output = self.fc3(output)  # (B, 2)
        return output


model = MyModel(node_embedding=embedding_gene, input_dim= input_dim, output_dim=2)
import torch.optim as optim

# Assume model is an instance of our RNN class
optimizer = optim.SGD(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
model = model.to(device)
criterion = criterion.to(device)


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    total_correct = 0
    total_instances = 0

    model.train()

    for X, y in iterator:
        optimizer.zero_grad()

        # Assuming your model's forward method automatically handles padding, then no need to pack sequence here
        predictions = model(X)

        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # Compute the number of correct predictions
        _, predicted_classes = predictions.max(dim=1)
        correct_predictions = (predicted_classes == y).float()  # Convert to float for summation
        total_correct += correct_predictions.sum().item()
        total_instances += y.size(0)

    epoch_acc = total_correct / total_instances

    return epoch_loss / len(iterator), epoch_acc


train_loss, train_acc = train(model, train_loader, optimizer, criterion)
def evaluate(model, iterator, criterion):
    epoch_loss = 0
    total_correct = 0
    total_instances = 0

    model.eval()

    with torch.no_grad():
        for X, y in iterator:
            predictions = model(X)

            loss = criterion(predictions, y)
            epoch_loss += loss.item()

            # Compute the number of correct predictions
            _, predicted_classes = predictions.max(dim=1)
            correct_predictions = (predicted_classes == y).float()  # Convert to float for summation
            total_correct += correct_predictions.sum().item()
            total_instances += y.size(0)

    epoch_acc = total_correct / total_instances
    return epoch_loss / len(iterator), epoch_acc
import time


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = elapsed_time - (elapsed_mins * 60)
    return elapsed_mins, elapsed_secs, elapsed_time
N_EPOCHS = 20

best_valid_loss = float('inf')

best_valid_acc = 0

elapsed_times = []

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, test_loader, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs, elapsed_time = epoch_time(start_time, end_time)
    elapsed_times.append(elapsed_time)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        best_valid_acc = valid_acc
        #torch.save(model.state_dict(), '../data/final_model.pt')

    if epoch % 1 == 0:
        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs:.3f}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

avg_elapsed_time = sum(elapsed_times) / len(elapsed_times)
print(f'Avg Epoch Time: {avg_elapsed_time:.3f}s')
print(f'Best Val. Loss: {best_valid_loss:.3f} | Best Val. Acc: {best_valid_acc * 100:.2f}%')