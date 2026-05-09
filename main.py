import sys
import os
import csv
import copy
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import dataset
from torch_geometric.loader import DataLoader
import torch.nn as nn
from torch_geometric.nn.models import GAT, GCN, MLP, GraphSAGE
from torch_geometric.nn.conv import APPNP
from sklearn.model_selection import StratifiedKFold
from pA_GAT_60_20_20 import train_graph, test_graph, WeightedGATGraphNet
from dataset_loader import prepare_planetoid_dataset, prepare_webkb_dataset, prepare_coauthor_dataset
from itertools import product 

WEBKB_DATASETS     = {'Texas', 'Cornell', 'Wisconsin'}
PLANETOID_DATASETS = {'Cora', 'Citeseer', 'Pubmed'}
COAUTHOR_DATASETS  = {'CS', 'Physics'}

class APPNPClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, dropout, K=10, alpha=0.1):
        super().__init__()

        self.mlp = MLP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            norm=None
        )
        self.prop = APPNP(K=K, alpha=alpha, dropout=dropout)

    def forward(self, x, edge_index):
        x = self.mlp(x)
        x = self.prop(x, edge_index)
        return x

# ------------------- Utility: set seeds -------------------
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_model(model_name, num_features, num_classes, hidden_dim, num_layers, dropout_rate):
    """Instantiates a fresh model for each fold."""
    if model_name == "PAGAT":
        return WeightedGATGraphNet(
            in_dim=num_features, hidden_dim=hidden_dim, num_layers=num_layers, 
            num_classes=num_classes, dropout_rate=dropout_rate, heads=4
        )
    elif model_name == "GraphSAGE":
        return GraphSAGE(
            in_channels=num_features,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            out_channels=num_classes,
            dropout=dropout_rate
        )
    elif model_name == "GAT":
        return GAT(
            in_channels=num_features,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            out_channels=num_classes,
            dropout=dropout_rate
        )
    elif model_name == "APPNP":
        return APPNPClassifier(
            in_channels=num_features,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            out_channels=num_classes,
            dropout=dropout_rate
        )
    elif model_name == "GCN":
        return GCN(
            in_channels=num_features,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            out_channels=num_classes,
            dropout=dropout_rate
        )
    else:
        raise ValueError(f"Unknown model {model_name}")

# ---------------- Experiment ---------------
def run_experiment(model_name, data, num_features, num_classes, device, hidden_dim, num_layers, dropout_rate, lr, weight_decay, num_runs):
    # We use all vall acc for hyperparameter tuning
    all_val_acc = []
    all_test_acc = []

    is_list = isinstance(data, list)

    if not is_list:
        current_data = data.to(device)

    for run_idx in range(num_runs):
        set_seed(42 + run_idx) 

        if is_list:
            current_data = data[run_idx][0].to(device)
         
        model = get_model(
                model_name=model_name,
                num_features=num_features,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout_rate=dropout_rate
                ).to(device) 

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) 
         
        best_val_acc = 0
        best_model_weights = None
        patience = 0 
        early_stop = 100

        for epoch in range(1,3001):
            train_graph(model, current_data, optimizer)

            val_acc, _ = test_graph(model, current_data)

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

        val_acc, test_acc = test_graph(model, current_data)
        print(f"[{model_name}] Run {run_idx+1}/10 | Test Acc: {test_acc:.4f}") 
        all_val_acc.append(val_acc)
        all_test_acc.append(test_acc)

    val_mean_acc = np.mean(all_val_acc)
    mean_acc = np.mean(all_test_acc)
    std_acc = np.std(all_test_acc)
    return val_mean_acc, mean_acc, std_acc

# ------------------- Main -------------------
if __name__ == '__main__':
    # Set seed
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Diffusion params for PAGAT
    diffusion_params = dict(alpha=0.5,base_anisotropy_c=1.3,beta=0.07)

    # Requires input dataset
    if len(sys.argv) > 1:
        input_dataset = sys.argv[1]
        if len(sys.argv) > 2:
            alpha = float(sys.argv[2])
            c = float(sys.argv[3])
            beta = float(sys.argv[4])
            diffusion_params = dict(alpha=alpha,base_anisotropy_c=1.3,beta=beta)
    else:
        print("Please input a dataset")
        exit(1)   

    hidden_dims = [64]
    num_layerss = [2,3,4]
    dropout_rates = [0,0.1,0.5]
    lrs = [1e-2,5e-4,1e-6]
    weight_decays = [0, 1e-4]
   
    hyperparams = list(product(hidden_dims, num_layerss, dropout_rates, lrs, weight_decays))

    generate_splits = True

    # Prepare datasets
    if input_dataset in WEBKB_DATASETS:
        dataset_unweighted = [prepare_webkb_dataset(input_dataset, split=split) for split in range(10)]
        dataset_weighted = [prepare_webkb_dataset(input_dataset, split=split, diffusion_params=diffusion_params) for split in range(10)]
        num_classes = dataset_unweighted[0][1]
        num_features = dataset_unweighted[0][2]
    elif input_dataset in COAUTHOR_DATASETS:
        dataset_unweighted, num_classes, num_features = prepare_coauthor_dataset(input_dataset) 
        dataset_weighted, _, _= prepare_coauthor_dataset(input_dataset, diffusion_params=diffusion_params)
    elif input_dataset in PLANETOID_DATASETS:
        dataset_unweighted, num_classes, num_features = prepare_planetoid_dataset(input_dataset)    
        dataset_weighted, _, _ = prepare_planetoid_dataset(input_dataset, diffusion_params=diffusion_params)    
    else:
        raise Exception("Invalid Dataset")


    # Models to test
    models = ("PAGAT", "GCN", "GAT", "APPNP", "GraphSAGE")

    # Result directory
    os.makedirs("./results", exist_ok=True)
    results_file = "./results/results.csv"

    # Write CSV header row
    #with open(results_file, 'w', newline='') as f:
    #    csv.writer(f).writerow(["Model", "Dataset", "Mean Acc", "Std Dev", "All Folds"])

    # Loop through models
    for model_name in models:
        # Route dataset
        if model_name == "PAGAT":
            current_dataset = dataset_weighted
        else:
            current_dataset = dataset_unweighted

        max_mean = -1
        saved_std = -1
        saved_mean = -1
        saved_hidden_dim = saved_num_layers = saved_dropout_rate = saved_lr = saved_weight_decay = -1
        for hidden_dim, num_layers, dropout_rate, lr, weight_decay in hyperparams:
            val_mean, test_mean, test_std = run_experiment(model_name, current_dataset, num_features, num_classes, device, hidden_dim, num_layers, dropout_rate, lr, weight_decay,10)
            if val_mean > max_mean: 
                max_mean = val_mean 
                saved_mean = test_mean
                saved_std = test_std
                saved_hidden_dim = hidden_dim
                saved_num_layers = num_layers
                saved_dropout_rate = dropout_rate
                saved_lr = lr
                saved_weight_decay = weight_decay
            
        with open(results_file, 'a', newline='') as f: 
            writer = csv.writer(f) 
            writer.writerow([model_name, input_dataset, saved_mean, saved_std, saved_hidden_dim, saved_num_layers, saved_dropout_rate, saved_lr, saved_weight_decay]) 
