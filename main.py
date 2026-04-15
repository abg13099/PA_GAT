import sys
import os
import csv
import copy
import random
import numpy as np
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GAT, GIN, GCN
from sklearn.model_selection import StratifiedKFold
from pA_GAT_60_20_20 import train_graph, test_graph, WeightedGATGraphNet
from data_loader import prepare_tu_dataset
from benchmark_diff_pool import DiffPool
from benchmark_gin import GIN0
from benchmark_graphsage import GraphSAGE
from benchmark_gcn import GCN

# ------------------- Utility: set seeds -------------------
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def k_fold(dataset, folds=10, seed=42):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    labels = [data.y.item() for data in dataset]

    # Get the 10 distinct fold buckets
    fold_indices = []
    for _, test_idx in skf.split(torch.zeros(len(dataset)), labels):
        fold_indices.append(test_idx)

    train_indices, val_indices, test_indices = [], [], []

    for i in range(folds):
        test_idx = fold_indices[i]
        val_idx = fold_indices[(i + 1) % folds]
        train_idx = np.concatenate([
            fold_indices[j] for j in range(folds) if j != i and j != (i + 1) % folds
        ])

        train_indices.append(torch.tensor(train_idx, dtype=torch.long))
        val_indices.append(torch.tensor(val_idx, dtype=torch.long))
        test_indices.append(torch.tensor(test_idx, dtype=torch.long))

    return train_indices, val_indices, test_indices

def run_model(build_model_fn, dataset, splits, num_features, num_classes, device, results_path, model_name):
    train_indices, test_indices = splits
    all_test_acc = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(zip(train_indices, test_indices)):
        set_seed(42 + fold_idx)
        
        train_dataset = [dataset[i] for i in train_idx]
        test_dataset  = [dataset[i] for i in test_idx]
        
        # val split carved from train
        val_size = int(0.1 * len(train_dataset))
        val_dataset   = train_dataset[:val_size]
        train_dataset = train_dataset[val_size:]
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
        test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)
        
        # only this line differs per model
        model = build_model_fn(num_features, num_classes).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        best_val_acc = 0
        patience = 0
        early_stop = 1000
        
        for epoch in range(1, 3001):
            if model_name == "DiffPool":
                train_graph_diffpool(model, train_loader, optimizer, device)
            else:
                train_graph(model, train_loader, optimizer, device)
            val_acc = test_graph(model, val_loader, device)  # val not test
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience = 0
            else:
                patience += 1
            if patience > early_stop:
                break
        
        test_acc = test_graph(model, test_loader, device)
        print(f"[{model_name}] Fold {fold_idx+1} | Test {test_acc:.4f}")
        all_test_acc.append(test_acc)
    
    mean_acc = float(np.mean(all_test_acc))
    std_acc  = float(np.std(all_test_acc))
    
    with open(results_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([model_name, mean_acc, std_acc, all_test_acc])
    
    return mean_acc, std_acc

def get_model(model_name, num_features, num_classes, max_nodes):
    """Instantiates a fresh model for each fold."""
    if model_name == "PAGAT":
        return WeightedGATGraphNet(
            in_dim=num_features, hidden_dim=64, num_layers=3, 
            num_classes=num_classes, dropout_rate=0.5, heads=4
            )
    elif model_name == "GRAPHSAGE":
        return GraphSAGE(
                num_features=num_features,
                num_classes=num_classes,
                num_layers=3,
                hidden=64
                )
    elif model_name == "GIN":
        return GIN0(
            num_features=num_features, 
            num_classes=num_classes, 
            num_layers=3, 
            hidden=64
            )
    elif model_name == "GCN":
        return GCN(
            num_features=num_features, 
            num_classes=num_classes, 
            num_layers=3, 
            hidden=64
            )
    elif model_name == "DiffPool":
        return DiffPool(
            in_dim=num_features, num_classes=num_classes,
            num_layers=4, hidden=64,
            max_nodes=max_nodes, ratio=0.25
            )
    else:
        raise ValueError(f"Unknown model {model_name}")

def train_graph_diffpool(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch=batch.batch)
        loss = F.nll_loss(out, batch.y) + model.link_loss + model.ent_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# ---------------- Experiment ---------------
def run_experiment(model_name, dataset, splits, num_features, num_classes, device, results_path, max_nodes):
    train_indices, val_indices, test_indices = splits
    all_test_acc = []

    for fold_idx in range(len(train_indices)):
        set_seed(42 + fold_idx) 
         
        train_dataset = [dataset[i] for i in train_indices[fold_idx]] 
        val_dataset   = [dataset[i] for i in val_indices[fold_idx]] 
        test_dataset  = [dataset[i] for i in test_indices[fold_idx]] 
         
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) 
        val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False) 
        test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

        model = get_model(model_name, num_features, num_classes, max_nodes).to(device) 
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) 
         
        best_val_acc = 0 
        best_model_weights = None
        patience = 0 
        early_stop = 100

        for epoch in range(1,3001):
            train_graph(model, train_loader, optimizer, device)

            val_acc = test_graph(model, val_loader, device)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience = 0
                best_model_weights = copy.deepcopy(model.state_dict())
            else:
                patience += 1

            if patience > early_stop: 
                print(f"      Early stopping at epoch {epoch}")
                break

        if best_model_weights is not None:
            model.load_state_dict(best_model_weights)

        test_acc = test_graph(model, test_loader, device)
        print(f"[{model_name}] Fold {fold_idx+1}/10 | Test Acc: {test_acc:.4f}") 
        all_test_acc.append(test_acc)

    mean_acc = np.mean(all_test_acc)
    std_acc = np.std(all_test_acc)

    with open(results_path, 'a', newline='') as f: 
        writer = csv.writer(f) 
        writer.writerow([model_name, input_dataset, mean_acc, std_acc, all_test_acc]) 
     
    return mean_acc, std_acc

# ------------------- Main -------------------
if __name__ == '__main__':
    # Set seed
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Requires input dataset
    if len(sys.argv) > 1:
        input_dataset = sys.argv[1]
    else:
        print("Please input a dataset")
        exit(1)   

    # Diffusion params for PAGAT
    diffusion_params = dict(alpha=3.24846,base_anisotropy_c=0.1668764,beta=0.1168307)

    # Prepare datasets
    dataset_unweighted, num_classes, num_features = prepare_tu_dataset(name=input_dataset)
    dataset_weighted, _, _ = prepare_tu_dataset(name=input_dataset, diffusion_params=diffusion_params)
    max_nodes_unweighted = max(data.num_nodes for data in dataset_unweighted)

    # Models to test
    models = ("PAGAT", "GIN", "GCN", "DiffPool", "GraphSage")

    # Result directory
    os.makedirs("./results", exist_ok=True)
    results_file = "./results/results.csv"

    # Write CSV header row
    with open(results_file, 'w', newline='') as f:
        csv.writer(f).writerow(["Model", "Dataset", "Mean Acc", "Std Dev", "All Folds"])

    # Loop through models
    for model_name in models:
        # Route dataset
        if model_name == "PAGAT":
            current_dataset = dataset_weighted
        else:
            print("A")
            current_dataset = dataset_unweighted

        splits = k_fold(current_dataset, folds=10, seed=42)

        mean, std = run_experiment(model_name, current_dataset, splits, num_features, num_classes, device, results_file, max_nodes_unweighted)

