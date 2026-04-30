import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import networkx as nx
from scipy.linalg import eigh
import torch
import torch.nn.functional as F
from torch import nn, optim
import torch_geometric # Added this line
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool 
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from community import community_louvain as co_louvain
import random
from torch_geometric.datasets import Planetoid, TUDataset, WebKB, Coauthor
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import SpectralClustering
from networkx.algorithms import community
from itertools import product
import sys
import os
import csv
import gc
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.samplers import CmaEsSampler

WEBKB_DATASETS     = {'Texas', 'Cornell', 'Wisconsin'}
PLANETOID_DATASETS = {'Cora', 'Citeseer', 'Pubmed'}
COAUTHOR_DATASETS  = {'CS', 'Physics'}

# ------------------- Utility: set seeds -------------------
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def extract_largest_connected_component(data):
    G_nx = to_networkx(data, to_undirected=True)

    if nx.is_connected(G_nx):
        print("Graph is connected. Proceeding with the full graph.")
        return data

    print(f"Graph with {data.num_nodes} nodes is disconnected. Extracting the largest connected component.")
    connected_components = list(nx.connected_components(G_nx))
    largest_component_nodes = max(connected_components, key=len)

    if len(largest_component_nodes) == data.num_nodes:
        print("Largest connected component contains all nodes. No change needed.")
        return data

    # Create a mapping from original node IDs to new contiguous node IDs (0 to N-1 for LCC)
    old_node_ids_in_lcc = sorted(list(largest_component_nodes))
    old_to_new_node_map = {old_id: new_id for new_id, old_id in enumerate(old_node_ids_in_lcc)}

    # Filter `data` attributes to keep only nodes in the largest connected component
    new_x = data.x[old_node_ids_in_lcc]
    new_y = data.y[old_node_ids_in_lcc]

    # Filter masks: Only keep mask entries for nodes in LCC, and re-index them
    new_train_mask = data.train_mask[old_node_ids_in_lcc]
    new_val_mask = data.val_mask[old_node_ids_in_lcc]
    new_test_mask = data.test_mask[old_node_ids_in_lcc]

    # Filter and re-index edge_index
    # Only keep edges where both source and target are in the LCC
    edge_index_list = data.edge_index.t().tolist() # Convert to list of [u, v] pairs
    new_edge_index_list = []
    for u_orig, v_orig in edge_index_list:
        if u_orig in largest_component_nodes and v_orig in largest_component_nodes:
            new_edge_index_list.append([old_to_new_node_map[u_orig], old_to_new_node_map[v_orig]])
    new_edge_index = torch.tensor(new_edge_index_list, dtype=torch.long).t().contiguous()

    # Create a new Data object for the largest connected component
    new_data = torch_geometric.data.Data(x=new_x, edge_index=new_edge_index, y=new_y)
    new_data.train_mask = new_train_mask
    new_data.val_mask = new_val_mask
    new_data.test_mask = new_test_mask
    new_data.num_nodes = len(old_node_ids_in_lcc) # Explicitly set num_nodes

    print(f"Reduced graph to largest connected component with {new_data.num_nodes} nodes.")
    return new_data

# ------------------- Build block operator D -------------------
def build_block_operator_D(G_nx, communities, alpha=0.2, base_anisotropy_c=1, beta=0.01):
    """
    Build block operator D with anisotropic intra-community diffusion
    AND Option 1 modulation for inter-community diffusion.
    """

    n = G_nx.number_of_nodes()
    nodes_list = list(G_nx.nodes())
    node_to_idx = {v: i for i, v in enumerate(nodes_list)}

    D = np.zeros((n, n), dtype=float)

    # Storage for Option 1
    comm_us = {}          # cid -> unstable eigenvector (block-local indexing)
    comm_u_maxabs = {}    # cid -> max |u|
    comm_lambda_u = {}    # cid -> lambda_unstable_rescaled
    node_pos_in_block = {}   # node -> (cid, local_idx)

    # ----------- Build each community block ------------
    for cid, block in enumerate(communities):
        nodes = list(block)
        idx = np.array([node_to_idx[v] for v in nodes], dtype=int)

        sub = G_nx.subgraph(nodes)
        L_block = nx.laplacian_matrix(sub, nodelist=nodes).toarray().astype(float)

        nb = L_block.shape[0]
        if nb == 0:
            continue

        if nb == 1:
            # 1-node block: no eigenvectors
            D_block = np.eye(1)
            u = np.array([[1.0]])     # dummy
            lam_unstable_rescaled = 1.0
        else:
            # ---- Compute spectrum ----
            vals, vecs = eigh(L_block)
            vals = np.real(vals)
            vecs = np.real(vecs)

            sort_idx = np.argsort(vals)
            vals = vals[sort_idx]
            vecs = vecs[:, sort_idx]

            # stable = second smallest eigenvector
            # unstable = largest eigenvector
            s = vecs[:, 1].reshape(-1, 1)
            u = vecs[:, -1].reshape(-1, 1)

            # (You can replace these by rescaling again if desired)
            lam_stable_rescaled = vals[1]
            lam_unstable_rescaled = vals[-1]

            # ---- Adaptive anisotropic scale ----
            #    Smaller in dense blocks, larger in sparse blocks
            if lam_unstable_rescaled < 1e-9:
                anisotropic_scale = 0.0
            else:
                anisotropic_scale = base_anisotropy_c / lam_unstable_rescaled


            # ---- Build anisotropic intra-community diffusion ----
            P = anisotropic_scale * (lam_unstable_rescaled * (u @ u.T) -
                                     lam_stable_rescaled * (s @ s.T))
            D_block = np.eye(nb) - alpha * L_block + P

        # ---- Store unstable direction info for Option 1 ----
        comm_us[cid] = u.flatten()
        comm_u_maxabs[cid] = np.max(np.abs(comm_us[cid])) if np.max(np.abs(comm_us[cid])) > 0 else 1.0
        comm_lambda_u[cid] = lam_unstable_rescaled

        for local_idx, node in enumerate(nodes):
            node_pos_in_block[node] = (cid, local_idx)

        # ---- Insert block into D ----
        for ii, vi in enumerate(idx):
            for jj, vj in enumerate(idx):
                D[vi, vj] = D_block[ii, jj]

    # ----------------- Inter-community edges (Option 1) -----------------

    # First compute Z (global max alignment factor)
    Z = 1.0
    max_num = 0.0
    for u_node, v_node in G_nx.edges():
        cid_u, pos_u = node_pos_in_block[u_node]
        cid_v, pos_v = node_pos_in_block[v_node]
        if cid_u == cid_v:
            continue

        # normalized |u|
        val_u = abs(comm_us[cid_u][pos_u]) / comm_u_maxabs[cid_u]
        val_v = abs(comm_us[cid_v][pos_v]) / comm_u_maxabs[cid_v]

        numerator = val_u * val_v * 0.5 * (comm_lambda_u[cid_u] + comm_lambda_u[cid_v])
        max_num = max(max_num, numerator)

    if max_num > 0:
        Z = max_num  # ensures w <= beta approximately

    # Now add weighted cross-community diffusion
    for u_node, v_node in G_nx.edges():
        cid_u, pos_u = node_pos_in_block[u_node]
        cid_v, pos_v = node_pos_in_block[v_node]

        if cid_u == cid_v:
            continue

        val_u = abs(comm_us[cid_u][pos_u]) / comm_u_maxabs[cid_u]
        val_v = abs(comm_us[cid_v][pos_v]) / comm_u_maxabs[cid_v]

        numerator = val_u * val_v * 0.5 * (comm_lambda_u[cid_u] + comm_lambda_u[cid_v])
        w = beta * (numerator / Z)

        iu = node_to_idx[u_node]
        iv = node_to_idx[v_node]
        D[iu, iv] += w
        D[iv, iu] += w   # symmetric

    return D, node_to_idx, nodes_list


# ------------------- Attach weights from D -------------------
def attach_weights_from_D(data, D, node_to_idx, nodes_list, clip_negative=True, add_self_loop=True):
    W = 0.5 * (D + D.T)
    if clip_negative:
        W[W < 0] = 0.0

    edge_index = data.edge_index.cpu().numpy()
    num_edges = edge_index.shape[1]
    edge_weights = np.zeros(num_edges, dtype=float)

    for eidx in range(num_edges):
        u = int(edge_index[0, eidx])
        v = int(edge_index[1, eidx])
        iu, iv = node_to_idx[u], node_to_idx[v]
        edge_weights[eidx] = W[iu, iv]

    data.edge_weight = torch.tensor(edge_weights, dtype=torch.float)

    if add_self_loop:
        u_list, v_list = data.edge_index[0].tolist(), data.edge_index[1].tolist()
        w_list = data.edge_weight.tolist()
        diag = np.diag(W)
        for node in range(data.num_nodes):
            exists = False
            for j, (uu, vv) in enumerate(zip(u_list, v_list)):
                if uu == node and vv == node:
                    w_list[j] = diag[node]
                    exists = True
                    break
            if not exists:
                u_list.append(node)
                v_list.append(node)
                w_list.append(float(diag[node]))
        data.edge_index = torch.tensor([u_list, v_list], dtype=torch.long)
        data.edge_weight = torch.tensor(w_list, dtype=torch.float)

    return data

# ------------------- Compute diffused Laplacian weights -------------------
def compute_diffused_laplacian_weights(data,
                                       alpha=0.1,
                                       base_anisotropy_c=1,
                                       beta=0.01,
                                       soft_positive=True):
    """
    Precompute anisotropic diffusion-derived edge weights for the entire graph data.
    Returns modified data and number of communities.
    """
    G_nx = to_networkx(data, to_undirected=True)
    partition = co_louvain.best_partition(G_nx, random_state=42)
    # Choose number of communities
    #k = 10

    # Returns an iterator of sets
    #comp = community.asyn_fluidc(G_nx, k=k)

    # Convert to partition dict
    #partition = {}
    #for comm_id, node_set in enumerate(comp):
    #    for node in node_set:
    #        partition[node] = comm_id

    # convert partition dict -> list of node sets
    communities_dict = {}
    for node, comm_id in partition.items():
        communities_dict.setdefault(comm_id, set()).add(node)
    communities_sets = list(communities_dict.values())
    num_communities = len(communities_sets)

    D, node_to_idx, nodes_list = build_block_operator_D(G_nx, communities_sets,
                                                       alpha=alpha,
                                                       base_anisotropy_c=base_anisotropy_c,
                                                       beta=beta)
    data = attach_weights_from_D(data, D, node_to_idx, nodes_list)
    return data, num_communities


class WeightedGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, dropout=0.0, add_self_loops=True, bias=True):
        super().__init__(aggr='add', node_dim=0)
        self.in_channels, self.out_channels, self.heads, self.concat = in_channels, out_channels, heads, concat
        self.dropout, self.add_self_loops = dropout, add_self_loops
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_l = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = nn.Parameter(torch.Tensor(1, heads, out_channels))
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        x_trans = self.lin(x)
        N = x_trans.size(0)
        x_trans = x_trans.view(N, self.heads, self.out_channels)

        if self.add_self_loops:
            self_loops = torch.arange(N, device=edge_index.device)
            self_loops = torch.stack([self_loops, self_loops], dim=0)
            edge_index = torch.cat([edge_index, self_loops], dim=1)
            if edge_weight is None:
                edge_weight = torch.ones(edge_index.size(1), device=x_trans.device)
            else:
                edge_weight = torch.cat([edge_weight, torch.ones(N, device=edge_weight.device)], dim=0)

        out = self.propagate(edge_index, x=x_trans, edge_weight=edge_weight, size=(N, N))
        out = out.view(N, self.heads * self.out_channels) if self.concat else out.mean(dim=1)
        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j, x_i, edge_index_i, edge_weight):
        el = (x_i * self.att_l).sum(dim=-1)
        er = (x_j * self.att_r).sum(dim=-1)
        e = F.leaky_relu(el + er, negative_slope=0.2)
        if edge_weight is not None:
            e = e * edge_weight.unsqueeze(-1)
        alpha = softmax(e, index=edge_index_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

# ------------------- Weighted GAT GraphNet -------------------
class WeightedGATGraphNet(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_layers=2, num_classes=7, dropout_rate=0.5, heads=4):
        super().__init__()

        self.convs, self.bns, self.dropout = nn.ModuleList(), nn.ModuleList(), nn.Dropout(dropout_rate)
        self.heads = heads

        for i in range(num_layers):
            in_ch = in_dim if i == 0 else hidden_dim * heads
            self.convs.append(WeightedGATConv(in_ch, hidden_dim, concat=True, heads=heads, dropout=dropout_rate))
            self.bns.append(nn.BatchNorm1d(hidden_dim * heads))

        self.classifier = nn.Linear(hidden_dim * heads, num_classes)

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is not None:
            edge_weight = softmax(edge_weight, edge_index[0])
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = self.bns[i](x)
            x = F.elu(x)
            x = self.dropout(x)
        return F.log_softmax(self.classifier(x), dim=1)

# ------------------- Training & Evaluation -------------------
def train_graph(model, data, optimizer):
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index, edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None)

    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()

    return loss 

def test_graph(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None)
        pred = out.argmax(dim=1)

        val_correct = pred[data.val_mask].eq(data.y[data.val_mask]).sum().item()
        val_total = data.val_mask.sum().item()
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        test_correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
        test_total = data.test_mask.sum().item()
        test_acc = test_correct / test_total if test_total > 0 else 0.0

    return val_acc, test_acc

def compute_structural_features(data):
    G = to_networkx(data, to_undirected = True)
    nodes = list(G.nodes())
    degrees = [G.degree(n) for n in nodes]
    clustering = list(nx.clustering(G).values())
    features = torch.tensor(
        [[d, c] for d, c in zip(degrees, clustering)],
        dtype=torch.float
    )
    data.x = features
    return data

def prepare_tu_dataset(name, root='data/TU', diffusion_params=None):

    dataset = TUDataset(root=root, name=name, use_node_attr=True)

    num_classes = dataset.num_classes
    num_features = dataset.num_features


    #Sometimes data is empty
    dataset = [data for data in dataset if data.y is not None]

    if dataset[0].x is None:
        print(f"No node features found for {name}, using constant features")
        dataset = [compute_structural_features(d) for d in dataset]
        num_features = 2

    if diffusion_params:
        processed=[]
        for data in dataset:
            try:
                data,_ = compute_diffused_laplacian_weights(data, **diffusion_params)
            except Exception as e:
                print(f"Error when computing diffused laplacian: {e}")
            processed.append(data)
        dataset = processed
    return dataset, num_classes, num_features 

def prepare_planetoid_dataset(name, root='data/Planetoid', diffusion_params=None):

    dataset = Planetoid(root=root, name=name)
    data = dataset[0]
    
    if diffusion_params:
        try:
            data,_ = compute_diffused_laplacian_weights(data, **diffusion_params)
        except Exception as e:
            print(f"Error computing diffusion params: {e}")
    return data, dataset.num_classes, dataset.num_features 

def prepare_webkb_dataset(name, root='data/WebKB', diffusion_params=None, split=0):
    
    dataset = WebKB(root=root, name=name)
    data = dataset[0]
    data = extract_largest_connected_component(data)

    data.train_mask = data.train_mask[:, split]
    data.val_mask   = data.val_mask[:, split]
    data.test_mask  = data.test_mask[:, split]

    if diffusion_params:
        try:
            data,_ = compute_diffused_laplacian_weights(data, **diffusion_params)
        except Exception as e:
            print(f"Error computing diffusion params: {e}")
    return data, dataset.num_classes, dataset.num_features 

def prepare_coauthor_dataset(name, root='data/Coauthor', diffusion_params=None):
    dataset = Coauthor(root=root, name=name)
    data = dataset[0]
    data = extract_largest_connected_component(data)

    num_nodes = data.num_nodes
    indices = np.random.permutation(num_nodes)

    #20/30/50 Split for Coautor Datasets
    train = int((0.2 * num_nodes))
    val = int((0.5 * num_nodes))

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    data.train_mask[indices[:train]] = True
    data.val_mask[indices[train:val]] = True
    data.test_mask[indices[val:]] = True
    
    if diffusion_params:
        try:
            data,_ = compute_diffused_laplacian_weights(data, **diffusion_params)
        except Exception as e:
            print(f"Error computing diffusion params: {e}")
    return data, dataset.num_classes, dataset.num_features 

def k_fold(dataset, folds=10, seed=42):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    labels = [data.y.item() for data in dataset]

    train_indices, test_indices = [], []
    for train_idx, test_idx in skf.split(torch.zeros(len(dataset)), labels):
        train_indices.append(torch.tensor(train_idx, dtype=torch.long))
        test_indices.append(torch.tensor(test_idx, dtype=torch.long))

    return train_indices, test_indices

def objective(trial, input_dataset, device):
    alpha = trial.suggest_float('alpha', 0.05, 5, log=True)
    c = trial.suggest_float('c', 0.1, 10.0, log=True)
    beta = trial.suggest_float('beta', 0.005, 3, log=True)

    diffusion_params = dict(alpha=alpha, base_anisotropy_c=c, beta=beta)

    try:
        if input_dataset in WEBKB_DATASETS:
            data, num_classes, num_features = prepare_webkb_dataset(
                input_dataset, diffusion_params=diffusion_params, split=0)
        elif input_dataset in COAUTHOR_DATASETS:
            data, num_classes, num_features = prepare_coauthor_dataset(
                input_dataset, diffusion_params=diffusion_params)
        elif input_dataset in PLANETOID_DATASETS:
            data, num_classes, num_features = prepare_planetoid_dataset(
                input_dataset, diffusion_params=diffusion_params)    
        else:
            raise Exception("Invalid Dataset")
    except Exception as e:
         print(f"Preprocessing failed: {e} skipping combination")
         raise optuna.exceptions.TrialPruned()

    print(f"Running experiment with: alpha: {alpha}, c: {c}, beta: {beta}")


    data = data.to(device)

    model = WeightedGATGraphNet(
        in_dim=num_features, hidden_dim=64, num_layers=2,
        num_classes=num_classes, dropout_rate=0.5, heads=8
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    test_accs = []

    best_val_acc = 0.0
    best_test_acc = 0.0
    patience_count = 0
    PATIENCE = 50

    for epoch in range(1,501):
        loss = train_graph(model, data, optimizer)
        
        val_acc, test_acc = test_graph(model, data)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= 50:
            break

        trial.report(best_val_acc, step = epoch)
        if trial.should_prune():
            print("Was gonna prune")
            del optimizer, data, model
            gc.collect()
            raise optuna.exceptions.TrialPruned()

    del optimizer, data, model
    gc.collect()
    return float(best_test_acc)

if __name__ == "__main__":
    set_seed(42)
    n_trials = 25
    if len(sys.argv) > 1:
        input_dataset = sys.argv[1]
        if len(sys.argv) > 2:
            n_trials = int(sys.argv[2])
    else:
        print("Requires input of dataset")
        exit(1)

    # Check current settings
    print(torch.get_num_threads())

    # Set PyTorch to use all 20 cores
    torch.set_num_threads(15)
    torch.set_num_interop_threads(15) 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('results', exist_ok=True)
    results_path = 'results/results.csv'
    study = optuna.create_study(
        direction='maximize',
        sampler=CmaEsSampler(seed=42, 
                                n_startup_trials=20,
                                sigma0=0.8  # smaller sigma = tighter search around that point
                             ),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )

    study.optimize(
        lambda trial: objective(trial, input_dataset, device),
        n_trials=n_trials,
        show_progress_bar=True
    )

    best = study.best_params
    diffusion_params = dict(
        alpha=best['alpha'],
        base_anisotropy_c=best['c'],
        beta=best['beta']
    )

    values = [t.value for t in study.trials if t.value is not None]
    sigma = np.std(values)

    importances = optuna.importance.get_param_importances(study)
    with open(results_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([input_dataset,best['alpha'], best['c'], best['beta'], sigma, importances])
