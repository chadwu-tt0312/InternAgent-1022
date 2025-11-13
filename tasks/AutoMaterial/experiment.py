"""
Baseline CGCNN (Crystal Graph Convolutional Neural Network) implementation
for formation energy prediction on MatBench dataset.

This is a baseline implementation that can be improved by InternAgent.
"""

import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Try to import matbench (optional, for dataset loading)
try:
    from matbench import MatbenchBenchmark
    MATBENCH_AVAILABLE = True
except ImportError:
    MATBENCH_AVAILABLE = False
    print("Warning: matbench not installed. Please install with: pip install matbench")


class CrystalGraphDataset(Dataset):
    """Dataset for crystal structures with periodic boundary conditions."""
    
    def __init__(self, structures, targets, max_neighbors=12, cutoff=8.0):
        """
        Args:
            structures: List of pymatgen Structure objects
            targets: List of formation energies (eV/atom)
            max_neighbors: Maximum number of neighbors per atom
            cutoff: Distance cutoff for neighbor finding (Angstrom)
        """
        self.structures = structures
        self.targets = targets
        self.max_neighbors = max_neighbors
        self.cutoff = cutoff
    
    def __len__(self):
        return len(self.structures)
    
    def __getitem__(self, idx):
        structure = self.structures[idx]
        target = self.targets[idx]
        
        # Build crystal graph with periodic neighbors
        node_features, edge_index, edge_attr = self._build_crystal_graph(structure)
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([target], dtype=torch.float32)
        )
        
        return data
    
    def _build_crystal_graph(self, structure):
        """Build graph from crystal structure with periodic boundary conditions."""
        num_atoms = len(structure)
        
        # Node features: atomic number (one-hot or embedding)
        atomic_numbers = [site.specie.Z for site in structure]
        node_features = torch.tensor(atomic_numbers, dtype=torch.long)
        
        # Find neighbors with periodic boundary conditions
        edge_index_list = []
        edge_attr_list = []
        
        for i in range(num_atoms):
            neighbors = structure.get_neighbors(structure[i], self.cutoff)
            
            # Sort by distance and take top max_neighbors
            neighbors = sorted(neighbors, key=lambda x: x[1])[:self.max_neighbors]
            
            for neighbor, distance, image in neighbors:
                j = neighbor.index
                # Add periodic image offset to edge attributes
                edge_index_list.append([i, j])
                edge_attr_list.append([distance, image[0], image[1], image[2]])
        
        if len(edge_index_list) == 0:
            # Fallback: connect to self if no neighbors found
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_attr = torch.tensor([[0.0, 0, 0, 0]], dtype=torch.float32)
        else:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
        
        return node_features, edge_index, edge_attr


class CGCNNConv(MessagePassing):
    """Crystal Graph Convolutional layer."""
    
    def __init__(self, in_channels, out_channels, num_edge_features=4):
        super(CGCNNConv, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Edge feature transformation
        self.edge_mlp = nn.Sequential(
            nn.Linear(num_edge_features, in_channels),
            nn.Softplus(),
            nn.Linear(in_channels, in_channels)
        )
        
        # Node feature transformation
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Softplus()
        )
    
    def forward(self, x, edge_index, edge_attr):
        # x: [N, in_channels]
        # edge_index: [2, E]
        # edge_attr: [E, num_edge_features]
        
        # Transform edge features
        edge_embedding = self.edge_mlp(edge_attr)
        
        # Message passing
        out = self.propagate(edge_index, x=x, edge_attr=edge_embedding)
        
        # Transform node features
        out = self.node_mlp(out)
        
        return out
    
    def message(self, x_i, x_j, edge_attr):
        # x_i: target node features
        # x_j: source node features
        # edge_attr: edge embeddings
        return edge_attr * x_j


class CGCNN(nn.Module):
    """Crystal Graph Convolutional Neural Network."""
    
    def __init__(self, 
                 atom_embedding_dim=64,
                 hidden_dim=128,
                 num_layers=3,
                 num_edge_features=4,
                 output_dim=1):
        super(CGCNN, self).__init__()
        
        # Atom embedding (maps atomic number to feature vector)
        # Assuming max atomic number is 118
        self.atom_embedding = nn.Embedding(119, atom_embedding_dim, padding_idx=0)
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.convs.append(CGCNNConv(atom_embedding_dim, hidden_dim, num_edge_features))
        
        for _ in range(num_layers - 1):
            self.convs.append(CGCNNConv(hidden_dim, hidden_dim, num_edge_features))
        
        # Global pooling and regression head
        self.pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Softplus(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, data):
        # data: Batch from DataLoader
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Embed atoms
        x = self.atom_embedding(x).squeeze()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Graph convolutions
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
        
        # Global pooling (mean pooling per graph)
        from torch_geometric.nn import global_mean_pool
        x = global_mean_pool(x, batch)
        
        # Regression
        x = self.pool(x)
        x = self.regressor(x)
        
        return x.squeeze()


def load_matbench_data(task_name='matbench_mp_e_form', use_subset=False, subset_size=1000):
    """Load MatBench dataset."""
    if not MATBENCH_AVAILABLE:
        raise ImportError("matbench is required. Install with: pip install matbench")
    
    mb = MatbenchBenchmark(autoload=False)
    task = mb.load_task(task_name)
    
    structures = []
    targets = []
    
    print(f"Loading {task_name} dataset...")
    
    for fold in task.folds:
        train_inputs, train_outputs = task.get_train_and_val_data(fold)
        
        if use_subset:
            train_inputs = train_inputs[:subset_size]
            train_outputs = train_outputs[:subset_size]
        
        for cif_str, target in zip(tqdm(train_inputs, desc="Parsing structures"), train_outputs):
            try:
                # Parse CIF string to Structure
                parser = CifParser.from_string(cif_str)
                structure = parser.get_structures()[0]
                structures.append(structure)
                targets.append(target)
            except Exception as e:
                print(f"Warning: Failed to parse structure: {e}")
                continue
    
    return structures, targets


def calculate_mae(y_true, y_pred):
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        pred = model(batch)
        loss = criterion(pred, batch.y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = batch.to(device)
            pred = model(batch)
            loss = criterion(pred, batch.y)
            
            total_loss += loss.item()
            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    mae = calculate_mae(all_targets, all_preds)
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, mae, all_preds, all_targets


def parse_args():
    parser = argparse.ArgumentParser(description='CGCNN for Formation Energy Prediction')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to preprocessed data (if not using MatBench)')
    parser.add_argument('--use_subset', action='store_true',
                       help='Use subset of data for faster testing')
    parser.add_argument('--subset_size', type=int, default=1000,
                       help='Size of subset if use_subset=True')
    
    # Model arguments
    parser.add_argument('--atom_embedding_dim', type=int, default=64,
                       help='Atom embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of graph convolution layers')
    parser.add_argument('--cutoff', type=float, default=8.0,
                       help='Distance cutoff for neighbors (Angstrom)')
    parser.add_argument('--max_neighbors', type=int, default=12,
                       help='Maximum number of neighbors per atom')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # I/O arguments
    parser.add_argument('--out_dir', type=str, default='./output',
                       help='Output directory')
    parser.add_argument('--save_model', action='store_true',
                       help='Save trained model')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    if args.data_path is None:
        # Use MatBench
        if not MATBENCH_AVAILABLE:
            raise ImportError("matbench is required. Install with: pip install matbench")
        
        structures, targets = load_matbench_data(
            use_subset=args.use_subset,
            subset_size=args.subset_size
        )
    else:
        # Load from file (implement if needed)
        raise NotImplementedError("Loading from file not implemented yet")
    
    # Split data (simple train/val split for baseline)
    split_idx = int(0.8 * len(structures))
    train_structures = structures[:split_idx]
    train_targets = targets[:split_idx]
    val_structures = structures[split_idx:]
    val_targets = targets[split_idx:]
    
    print(f"Train samples: {len(train_structures)}, Val samples: {len(val_structures)}")
    
    # Create datasets
    train_dataset = CrystalGraphDataset(
        train_structures, train_targets,
        max_neighbors=args.max_neighbors,
        cutoff=args.cutoff
    )
    val_dataset = CrystalGraphDataset(
        val_structures, val_targets,
        max_neighbors=args.max_neighbors,
        cutoff=args.cutoff
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0
    )
    
    # Create model
    model = CGCNN(
        atom_embedding_dim=args.atom_embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_edge_features=4,  # distance + 3 periodic image offsets
        output_dim=1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Training loop
    best_mae = float('inf')
    best_epoch = 0
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_mae, val_preds, val_targets = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f} eV/atom")
        
        # Save best model
        if val_mae < best_mae:
            best_mae = val_mae
            best_epoch = epoch + 1
            if args.save_model:
                torch.save(model.state_dict(), os.path.join(args.out_dir, 'best_model.pt'))
    
    print(f"\nBest MAE: {best_mae:.6f} eV/atom at epoch {best_epoch}")
    
    # Final evaluation on validation set
    print("\nFinal evaluation...")
    final_loss, final_mae, final_preds, final_targets = evaluate(model, val_loader, criterion, device)
    
    # Save results
    final_infos = {
        "AutoMaterial": {
            "means": {
                "MAE": f"{final_mae:.6f}"
            }
        }
    }
    
    with open(os.path.join(args.out_dir, 'final_info.json'), 'w') as f:
        json.dump(final_infos, f, indent=4)
    
    print(f"\nResults saved to {args.out_dir}/final_info.json")
    print(f"Final MAE: {final_mae:.6f} eV/atom")


if __name__ == '__main__':
    main()

