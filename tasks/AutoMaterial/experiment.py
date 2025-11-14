"""AutoMaterial baseline: CGCNN training for crystal formation energy prediction.

This implementation follows the task description in README.md/中文說明.md:
    - Primary objective: predict formation energy (eV/atom) from crystal structures.
    - Baseline model: Crystal Graph Convolutional Neural Network (CGCNN).
    - Dataset priority: MatBench (matbench_mp_e_form). When the official
      package is unavailable, we generate physically-inspired synthetic
      crystal structures via pymatgen so that the CGCNN pipeline remains
      executable in offline environments.

Usage examples:
    # Full MatBench run (requires matbench/scikit-learn wheels)
    python experiment.py --dataset_source matbench --epochs 100

    # Deterministic synthetic dataset (default when MatBench is missing)
    python experiment.py --dataset_source synthetic --epochs 30 --use_subset --subset_size 1024
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
from pymatgen.core import Element, Lattice, Structure
from tqdm import tqdm

try:
    from matbench import MatbenchBenchmark

    MATBENCH_AVAILABLE = True
except ImportError:
    MATBENCH_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Data definitions
# --------------------------------------------------------------------------- #


class CrystalGraphDataset(Dataset):
    """Wrap a list of pymatgen Structure objects into PyG Data instances."""

    def __init__(self, structures, targets, max_neighbors=12, cutoff=8.0):
        self.structures = structures
        self.targets = targets
        self.max_neighbors = max_neighbors
        self.cutoff = cutoff

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        structure = self.structures[idx]
        target = self.targets[idx]
        node_features, edge_index, edge_attr = self._build_crystal_graph(structure)
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([target], dtype=torch.float32),
        )
        return data

    def _build_crystal_graph(self, structure):
        num_atoms = len(structure)
        atomic_numbers = [site.specie.Z for site in structure]
        node_features = torch.tensor(atomic_numbers, dtype=torch.long)

        edge_index_list = []
        edge_attr_list = []
        for i in range(num_atoms):
            neighbors = structure.get_neighbors(structure[i], self.cutoff)
            neighbors = sorted(neighbors, key=lambda x: x.nn_distance)[: self.max_neighbors]
            for neighbor in neighbors:
                image = getattr(neighbor, "image", (0, 0, 0))
                edge_index_list.append([i, neighbor.index])
                edge_attr_list.append([neighbor.nn_distance, image[0], image[1], image[2]])

        if not edge_index_list:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_attr = torch.tensor([[0.0, 0, 0, 0]], dtype=torch.float32)
        else:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)

        return node_features, edge_index, edge_attr


# --------------------------------------------------------------------------- #
# CGCNN model
# --------------------------------------------------------------------------- #


class CGCNNConv(MessagePassing):
    """Single CGCNN convolution block."""

    def __init__(self, in_channels, out_channels, num_edge_features=4):
        super().__init__(aggr="add")
        self.edge_mlp = nn.Sequential(
            nn.Linear(num_edge_features, in_channels),
            nn.Softplus(),
            nn.Linear(in_channels, in_channels),
        )
        self.node_mlp = nn.Sequential(nn.Linear(in_channels, out_channels), nn.Softplus())

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_mlp(edge_attr)
        out = self.propagate(edge_index, x=x, edge_attr=edge_embedding)
        return self.node_mlp(out)

    def message(self, x_j, edge_attr):
        return edge_attr * x_j


class CGCNN(nn.Module):
    """Minimal CGCNN regressor with global mean pooling."""

    def __init__(
        self,
        atom_embedding_dim=92,
        hidden_dim=256,
        num_layers=4,
        num_edge_features=4,
        output_dim=1,
    ):
        super().__init__()
        self.atom_embedding = nn.Embedding(119, atom_embedding_dim, padding_idx=0)

        self.convs = nn.ModuleList()
        self.convs.append(CGCNNConv(atom_embedding_dim, hidden_dim, num_edge_features))
        for _ in range(num_layers - 1):
            self.convs.append(CGCNNConv(hidden_dim, hidden_dim, num_edge_features))

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Softplus(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.atom_embedding(x).squeeze()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)
        return self.regressor(x).squeeze()


# --------------------------------------------------------------------------- #
# Synthetic crystal generator (pymatgen-based) for offline execution
# --------------------------------------------------------------------------- #


ELEMENT_POOL = [
    Element("Li"),
    Element("Be"),
    Element("B"),
    Element("C"),
    Element("N"),
    Element("O"),
    Element("Na"),
    Element("Mg"),
    Element("Al"),
    Element("Si"),
    Element("P"),
    Element("S"),
    Element("K"),
    Element("Ca"),
    Element("Ti"),
    Element("V"),
    Element("Cr"),
    Element("Mn"),
    Element("Fe"),
    Element("Co"),
    Element("Ni"),
    Element("Cu"),
    Element("Zn"),
]


@dataclass
class SyntheticCrystal:
    structure: Structure
    target: float


def random_structure(seed_rng: random.Random) -> SyntheticCrystal:
    num_atoms = seed_rng.randint(3, 9)
    a, b, c = [seed_rng.uniform(3.0, 9.5) for _ in range(3)]
    alpha = seed_rng.uniform(75, 120)
    beta = seed_rng.uniform(75, 125)
    gamma = seed_rng.uniform(60, 120)
    lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)

    species = [seed_rng.choice(ELEMENT_POOL).symbol for _ in range(num_atoms)]
    frac_coords = [[seed_rng.random(), seed_rng.random(), seed_rng.random()] for _ in range(num_atoms)]
    structure = Structure(lattice, species, frac_coords, coords_are_cartesian=False)

    target = synthetic_energy(structure, seed_rng)
    return SyntheticCrystal(structure=structure, target=target)


def synthetic_energy(structure: Structure, rng: random.Random) -> float:
    comp = structure.composition
    num_atoms = comp.num_atoms
    avg_Z = sum(el.Z * amt for el, amt in comp.items()) / num_atoms
    avg_en = sum((el.X or 1.5) * amt for el, amt in comp.items()) / num_atoms
    avg_mass = sum(el.atomic_mass * amt for el, amt in comp.items()) / num_atoms
    volume_per_atom = structure.volume / num_atoms
    density = structure.density
    a, b, c = structure.lattice.abc
    anisotropy = (max(a, b, c) - min(a, b, c)) / max(1.0, min(a, b, c))

    energy = (
        -0.22 * avg_Z
        + 0.58 * avg_en
        - 0.08 * (density / 5.0)
        + 0.03 * (volume_per_atom / 10.0)
        - 0.04 * (avg_mass / 50.0)
        + 0.12 * anisotropy
    )
    energy += rng.gauss(0, 0.02)
    return float(energy)


def generate_synthetic_dataset(total_samples: int, seed: int) -> Tuple[List[Structure], List[float]]:
    rng = random.Random(seed)
    dataset = [random_structure(rng) for _ in range(total_samples)]
    structures = [item.structure for item in dataset]
    targets = [item.target for item in dataset]
    return structures, targets


# --------------------------------------------------------------------------- #
# Data loading helpers (MatBench or synthetic)
# --------------------------------------------------------------------------- #


def load_matbench_data(task_name="matbench_mp_e_form", use_subset=False, subset_size=1000):
    if not MATBENCH_AVAILABLE:
        raise RuntimeError("matbench is not installed. Install it or use --dataset_source synthetic.")
    mb = MatbenchBenchmark(autoload=False)
    task = mb.tasks_map[task_name]
    task.load()
    structures: List[Structure] = []
    targets: List[float] = []
    for fold in task.folds:
        train_inputs, train_outputs = task.get_train_and_val_data(fold)
        if use_subset:
            train_inputs = train_inputs[:subset_size]
            train_outputs = train_outputs[:subset_size]
        iterator = zip(train_inputs, train_outputs)
        iterator = tqdm(iterator, total=len(train_inputs), desc=f"Loading {task_name} (fold {fold})")
        for structure, target in iterator:
            structures.append(structure)
            targets.append(float(target))
    return structures, targets


def load_structures_from_json(path: str) -> Tuple[List[Structure], List[float]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    structures = []
    targets = []
    for entry in payload:
        if "lattice" not in entry or "species" not in entry or "frac_coords" not in entry:
            raise ValueError("Custom dataset entries must include lattice, species, frac_coords, and target.")
        lattice = Lattice(entry["lattice"])
        structure = Structure(lattice, entry["species"], entry["frac_coords"], coords_are_cartesian=False)
        structures.append(structure)
        targets.append(float(entry["target"]))
    return structures, targets


def prepare_structures(args) -> Tuple[List[Structure], List[float], str]:
    if args.data_path:
        structures, targets = load_structures_from_json(args.data_path)
        return structures, targets, f"custom-json:{args.data_path}"

    dataset_source = args.dataset_source
    if dataset_source == "auto":
        dataset_source = "matbench" if MATBENCH_AVAILABLE else "synthetic"

    if dataset_source == "matbench":
        structures, targets = load_matbench_data(
            use_subset=args.use_subset,
            subset_size=args.subset_size,
        )
        return structures, targets, "MatBench-mp_e_form"

    if dataset_source == "synthetic":
        total = args.synthetic_size
        if args.use_subset:
            total = min(total, args.subset_size)
        structures, targets = generate_synthetic_dataset(total, args.seed)
        return structures, targets, f"synthetic-{total}"

    raise ValueError(f"Unknown dataset_source {args.dataset_source}")


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #


def calculate_mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(dataloader))


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds = []
    targets = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            batch = batch.to(device)
            pred = model(batch)
            loss = criterion(pred, batch.y)
            total_loss += loss.item()
            preds.extend(pred.detach().cpu().view(-1).tolist())
            targets.extend(batch.y.detach().cpu().view(-1).tolist())
    mae = calculate_mae(targets, preds)
    return total_loss / max(1, len(dataloader)), mae


def split_dataset(structures, targets, val_ratio, seed):
    assert len(structures) == len(targets)
    total = len(structures)
    if total < 2:
        raise ValueError("Need at least 2 samples")
    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = max(1, int(total * val_ratio))
    train_idx = indices[val_size:]
    val_idx = indices[:val_size]

    def slice_list(items, idxs):
        return [items[i] for i in idxs]

    return (
        slice_list(structures, train_idx),
        slice_list(targets, train_idx),
        slice_list(structures, val_idx),
        slice_list(targets, val_idx),
    )


def write_final_info(out_dir: str, mae: float):
    payload = {"AutoMaterial": {"means": {"MAE": f"{mae:.6f}"}}}
    with open(os.path.join(out_dir, "final_info.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args():
    parser = argparse.ArgumentParser(description="AutoMaterial CGCNN experiment")
    parser.add_argument("--data_path", type=str, default=None, help="Optional JSON file with custom structures")
    parser.add_argument(
        "--dataset_source",
        type=str,
        default="auto",
        choices=["auto", "matbench", "synthetic"],
        help="Preferred dataset source",
    )
    parser.add_argument("--use_subset", action="store_true", help="Slice dataset for faster debugging")
    parser.add_argument("--subset_size", type=int, default=2048, help="Subset size when --use_subset is enabled")
    parser.add_argument("--synthetic_size", type=int, default=4096, help="Total synthetic samples when needed")
    parser.add_argument("--atom_embedding_dim", type=int, default=92)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--cutoff", type=float, default=8.0)
    parser.add_argument("--max_neighbors", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--out_dir", type=str, default="./output")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "auto"], help="Kept for backward compatibility")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode not in {"full", "auto"}:
        raise ValueError("Only full/auto modes are supported.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    structures, targets, dataset_label = prepare_structures(args)
    print(f"Dataset source: {dataset_label} | total samples = {len(structures)}")

    train_structures, train_targets, val_structures, val_targets = split_dataset(
        structures,
        targets,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"Train samples: {len(train_structures)}, Val samples: {len(val_structures)}")

    train_dataset = CrystalGraphDataset(train_structures, train_targets, args.max_neighbors, args.cutoff)
    val_dataset = CrystalGraphDataset(val_structures, val_targets, args.max_neighbors, args.cutoff)

    train_loader = PyGDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = PyGDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = CGCNN(
        atom_embedding_dim=args.atom_embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_edge_features=4,
        output_dim=1,
    ).to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    best_mae = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val   Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f}")
        if val_mae < best_mae:
            best_mae = val_mae
            best_state = model.state_dict()

    if best_state and args.save_model:
        torch.save(best_state, os.path.join(args.out_dir, "best_model.pt"))

    write_final_info(args.out_dir, best_mae)
    print(f"\nBest validation MAE: {best_mae:.6f} eV/atom")
    print(f"Results stored in {os.path.join(args.out_dir, 'final_info.json')}")


if __name__ == "__main__":
    main()
