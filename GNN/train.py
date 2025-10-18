import argparse

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from dataset import AMLtoGraph
from graphsmote import GraphSmote
from model import GAT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GNN for AML detection.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="data",
        help="Path to the AMLtoGraph dataset root (contains raw/processed folders).",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Mini-batch size for NeighborLoader."
    )
    parser.add_argument(
        "--num-neighbors",
        type=int,
        nargs="+",
        default=[30, 30],
        help="Number of neighbours to sample per layer.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Optimizer learning rate."
    )
    parser.add_argument(
        "--use-graphsmote",
        action="store_true",
        help="Enable GraphSMOTE augmentation on the training nodes.",
    )
    parser.add_argument(
        "--graphsmote-target-ratio",
        type=float,
        default=1.0,
        help="Desired minority/majority ratio after applying GraphSMOTE.",
    )
    parser.add_argument(
        "--graphsmote-feature-k",
        type=int,
        default=5,
        help="k-nearest neighbours used by SMOTE in feature space.",
    )
    parser.add_argument(
        "--graphsmote-edge-k",
        type=int,
        default=5,
        help="Number of neighbours to connect each synthetic node to.",
    )
    parser.add_argument(
        "--graphsmote-directed",
        action="store_true",
        help="Keep synthetic edges directed (no reverse edge will be added).",
    )
    return parser.parse_args()


def build_model(data: Data) -> GAT:
    model = GAT(
        in_channels=data.num_features, hidden_channels=16, out_channels=1, heads=8
    )
    return model


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = AMLtoGraph(args.dataset_root)
    data = dataset[0]

    model = build_model(data).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    split = T.RandomNodeSplit(split="train_rest", num_val=0.1, num_test=0.0)
    data = split(data)

    if args.use_graphsmote:
        augmentor = GraphSmote(
            target_ratio=args.graphsmote_target_ratio,
            feature_k=args.graphsmote_feature_k,
            edge_k=args.graphsmote_edge_k,
            bidirectional_edges=not args.graphsmote_directed,
        )
        data = augmentor(data, mask=data.train_mask)

    train_loader = NeighborLoader(
        data,
        num_neighbors=args.num_neighbors,
        batch_size=args.batch_size,
        input_nodes=data.train_mask,
    )

    val_loader = NeighborLoader(
        data,
        num_neighbors=args.num_neighbors,
        batch_size=args.batch_size,
        input_nodes=data.val_mask,
    )

    for epoch in range(args.epochs):
        total_loss = 0.0
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = criterion(pred, batch.y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += float(loss)

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"Epoch: {epoch:03d}, Loss: {total_loss:.4f}")
            evaluate(model, val_loader, device)


def evaluate(model: GAT, loader: NeighborLoader, device: torch.device) -> None:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            preds = (logits >= 0.5).to(torch.long).view(-1)
            labels = batch.y.to(torch.long)
            total += labels.numel()
            correct += (preds == labels).sum().item()
    accuracy = correct / total if total > 0 else 0.0
    print(f"Validation accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()

