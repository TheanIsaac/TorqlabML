"""End-to-end PyTorch training script for injured vs uninjured classification."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .datasets import build_feature_table


class TabularDataset(Dataset):
    """Simple dataset wrapper around numpy feature and label arrays."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.from_numpy(features.astype(np.float32, copy=False))
        self.labels = torch.from_numpy(labels.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


@dataclass
class PreprocessingStats:
    medians: pd.Series
    means: pd.Series
    stds: pd.Series


@dataclass
class ModelSpec:
    build_fn: Callable[[], nn.Module]
    lr: float = 1e-3
    weight_decay: float = 0.0


def compute_preprocessing_stats(features: pd.DataFrame) -> PreprocessingStats:
    """Calculate imputation and scaling statistics from training data."""

    medians = features.median()
    imputed = features.fillna(medians)
    means = imputed.mean()
    stds = imputed.std(ddof=0).replace(0.0, 1.0)
    return PreprocessingStats(medians=medians, means=means, stds=stds)


def apply_preprocessing(features: pd.DataFrame, stats: PreprocessingStats) -> np.ndarray:
    """Apply median imputation and standardization to feature table."""

    imputed = features.fillna(stats.medians)
    scaled = (imputed - stats.means) / stats.stds
    return scaled.to_numpy(dtype=np.float32, copy=False)


def create_data_loader(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TabularDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def compute_positive_class_weight(labels: np.ndarray) -> float:
    positives = float(labels.sum())
    negatives = float(len(labels) - positives)
    if positives <= 0 or negatives <= 0:
        return 1.0
    return negatives / positives


class LogisticRegressionNet(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Iterable[int],
        dropout: float,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


def build_model_registry(input_dim: int, random_state: int) -> Dict[str, ModelSpec]:
    """Create the collection of PyTorch models to benchmark."""

    torch.manual_seed(random_state)
    return {
        "logistic_regression": ModelSpec(
            build_fn=lambda: LogisticRegressionNet(input_dim),
            lr=1e-2,
            weight_decay=0.0,
        ),
        "mlp_small": ModelSpec(
            build_fn=lambda: MLPClassifier(input_dim, hidden_sizes=(64,), dropout=0.1),
            lr=3e-3,
            weight_decay=1e-4,
        ),
        "mlp_medium": ModelSpec(
            build_fn=lambda: MLPClassifier(input_dim, hidden_sizes=(128, 64), dropout=0.2),
            lr=1e-3,
            weight_decay=5e-4,
        ),
    }


def _ensure_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        return logits.unsqueeze(1)
    return logits


def _train_single_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    *,
    epochs: int,
    patience: int,
    device: torch.device,
) -> None:
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for _ in range(max(epochs, 1)):
        model.train()
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            logits = _ensure_logits(model(features))
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        val_loss = _evaluate_loss(model, val_loader, criterion, device)
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)


def _evaluate_loss(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device).unsqueeze(1)
            logits = _ensure_logits(model(features))
            loss = criterion(logits, labels)
            total_loss += loss.item() * features.size(0)
            total_samples += features.size(0)
    return total_loss / max(total_samples, 1)


def _predict(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_probs: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            logits = _ensure_logits(model(features))
            logits = logits.squeeze(1)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).to(torch.int64)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_prob = np.concatenate(all_probs)
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    return y_true, y_pred, y_prob


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    y_true_int = y_true.astype(int)
    y_pred_int = y_pred.astype(int)
    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(y_true_int, y_pred_int),
        "precision": precision_score(y_true_int, y_pred_int, zero_division=0),
        "recall": recall_score(y_true_int, y_pred_int, zero_division=0),
        "f1": f1_score(y_true_int, y_pred_int, zero_division=0),
    }

    if len(np.unique(y_true_int)) == 2:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true_int, y_prob)
        except ValueError:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def _run_cross_validation(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    cv_folds: int,
    random_state: int,
    device: torch.device,
    batch_size: int,
    epochs: int,
    patience: int,
) -> Dict[str, float]:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    fold_metrics: List[Dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        fold_stats = compute_preprocessing_stats(X_train.iloc[train_idx])
        fold_X_train = apply_preprocessing(X_train.iloc[train_idx], fold_stats)
        fold_X_val = apply_preprocessing(X_train.iloc[val_idx], fold_stats)
        fold_y_train = y_train.iloc[train_idx].to_numpy(dtype=np.float32, copy=False)
        fold_y_val = y_train.iloc[val_idx].to_numpy(dtype=np.float32, copy=False)

        train_loader = create_data_loader(
            fold_X_train,
            fold_y_train,
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = create_data_loader(
            fold_X_val,
            fold_y_val,
            batch_size=batch_size,
            shuffle=False,
        )

        torch.manual_seed(random_state + fold_idx)
        np.random.seed(random_state + fold_idx)
        model = spec.build_fn().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=spec.lr, weight_decay=spec.weight_decay)
        pos_weight_value = compute_positive_class_weight(fold_y_train)
        pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        _train_single_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            epochs=epochs,
            patience=patience,
            device=device,
        )

        y_true, y_pred, y_prob = _predict(model, val_loader, device)
        fold_metrics.append(_compute_metrics(y_true, y_pred, y_prob))

    if not fold_metrics:
        return {}

    aggregated: Dict[str, float] = {}
    for key in fold_metrics[0].keys():
        values = [metrics[key] for metrics in fold_metrics if not np.isnan(metrics[key])]
        aggregated[f"cv_{key}"] = float(np.mean(values)) if values else float("nan")
    return aggregated


def evaluate_models(
    model_registry: Dict[str, ModelSpec],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    cv_folds: int,
    random_state: int,
    device: torch.device,
    batch_size: int,
    epochs: int,
    patience: int,
    val_size: float,
    verbose_report: bool = False,
) -> pd.DataFrame:
    """Fit each PyTorch model, evaluate on the hold-out test set, and optionally run CV."""

    stats = compute_preprocessing_stats(X_train)
    X_train_np = apply_preprocessing(X_train, stats)
    X_test_np = apply_preprocessing(X_test, stats)
    y_train_np = y_train.to_numpy(dtype=np.float32, copy=False)
    y_test_np = y_test.to_numpy(dtype=np.float32, copy=False)

    num_samples = len(X_train_np)
    stratify_labels = y_train_np if len(np.unique(y_train_np)) > 1 else None
    if val_size <= 0 or num_samples < 2:
        train_idx = np.arange(num_samples)
        val_idx = np.arange(num_samples)
    else:
        try:
            train_idx, val_idx = train_test_split(
                np.arange(num_samples),
                test_size=val_size,
                random_state=random_state,
                stratify=stratify_labels,
            )
        except ValueError:
            train_idx = np.arange(num_samples)
            val_idx = np.arange(num_samples)

    train_loader = create_data_loader(
        X_train_np[train_idx],
        y_train_np[train_idx],
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = create_data_loader(
        X_train_np[val_idx],
        y_train_np[val_idx],
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = create_data_loader(
        X_test_np,
        y_test_np,
        batch_size=batch_size,
        shuffle=False,
    )

    results: List[Dict[str, float]] = []
    for name, spec in model_registry.items():
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        model = spec.build_fn().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=spec.lr, weight_decay=spec.weight_decay)
        pos_weight_value = compute_positive_class_weight(y_train_np[train_idx])
        pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        _train_single_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            epochs=epochs,
            patience=patience,
            device=device,
        )

        y_true, y_pred, y_prob = _predict(model, test_loader, device)
        metrics = _compute_metrics(y_true, y_pred, y_prob)
        metrics["model"] = name

        if cv_folds and cv_folds > 1:
            try:
                metrics.update(
                    _run_cross_validation(
                        spec,
                        X_train,
                        y_train,
                        cv_folds=cv_folds,
                        random_state=random_state,
                        device=device,
                        batch_size=batch_size,
                        epochs=epochs,
                        patience=patience,
                    )
                )
            except ValueError as err:
                print(f"Skipping cross-validation for {name}: {err}")

        results.append(metrics)

        if verbose_report:
            print("\n=== Classification report:", name, "===")
            print(classification_report(y_true.astype(int), y_pred.astype(int), digits=3))

    summary = pd.DataFrame(results).set_index("model")
    summary.sort_values(by="f1", ascending=False, inplace=True)
    return summary


def parse_args(args: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("../Data/Splits/manifest.csv"),
        help="Path to the manifest CSV/JSON produced by splits.py",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction for the test split")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="Number of cross-validation folds to run on the training set",
    )
    parser.add_argument("--time-column", type=str, default=None, help="Explicit time column name")
    parser.add_argument(
        "--torque-column", type=str, default=None, help="Explicit torque column name"
    )
    parser.add_argument(
        "--start-threshold",
        type=float,
        default=0.05,
        help="Onset detection threshold as percent of peak torque",
    )
    parser.add_argument(
        "--end-threshold",
        type=float,
        default=0.25,
        help="End detection threshold as percent of peak torque",
    )
    parser.add_argument(
        "--plateau-window-ms",
        type=int,
        default=500,
        help="Plateau window size in milliseconds",
    )
    parser.add_argument(
        "--plateau-step-ms",
        type=int,
        default=25,
        help="Sliding window step size in milliseconds",
    )
    parser.add_argument(
        "--plateau-cov-cap",
        type=float,
        default=None,
        help="Optional cap on plateau coefficient of variation",
    )
    parser.add_argument(
        "--save-features",
        type=Path,
        default=None,
        help="Optional path to write the feature table (CSV)",
    )
    parser.add_argument(
        "--save-metrics",
        type=Path,
        default=None,
        help="Optional path to write the model comparison table (CSV)",
    )
    parser.add_argument(
        "--verbose-report",
        action="store_true",
        help="Print a classification report for each model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size for PyTorch training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of epochs for each model",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience in epochs",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Fraction of training data reserved for validation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device to use (e.g. 'cuda', 'cpu'). Defaults to CUDA if available.",
    )
    return parser.parse_args(args)


def main(cli_args: Optional[Iterable[str]] = None) -> pd.DataFrame:
    args = parse_args(cli_args)

    X, y, meta = build_feature_table(
        args.manifest,
        sampling_rate_hz=1000,
        time_column=args.time_column,
        torque_column=args.torque_column,
        start_threshold_pct_of_peak=args.start_threshold,
        end_threshold_pct_of_peak=args.end_threshold,
        plateau_window_ms=args.plateau_window_ms,
        plateau_step_ms=args.plateau_step_ms,
        plateau_cov_cap=args.plateau_cov_cap,
    )

    if args.save_features:
        features_out = Path(args.save_features)
        features_out.parent.mkdir(parents=True, exist_ok=True)
        feature_table = X.copy()
        feature_table.insert(0, "injured", y.values)
        feature_table.to_csv(features_out, index=False)
        print(f"Saved feature table to {features_out}")

    stratify = y if len(np.unique(y)) > 1 else None
    split = train_test_split(
        X,
        y,
        meta,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify,
    )
    X_train, X_test, y_train, y_test, meta_train, meta_test = split

    print(
        f"Loaded {len(X)} samples with {X.shape[1]} features. "
        f"Train/Test sizes: {len(X_train)}/{len(X_test)}"
    )
    print("Label distribution (overall):")
    print(y.value_counts())
    if not meta_test.empty:
        print("\nSample test files:")
        print(meta_test.head()["path"].to_string(index=False))

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using torch device: {device}")

    models = build_model_registry(X.shape[1], args.random_state)
    summary = evaluate_models(
        models,
        X_train,
        y_train,
        X_test,
        y_test,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        device=device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        val_size=args.val_size,
        verbose_report=args.verbose_report,
    )

    print("\nModel comparison (sorted by F1):")
    print(summary)

    if args.save_metrics:
        metrics_out = Path(args.save_metrics)
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(metrics_out)
        print(f"Saved metrics summary to {metrics_out}")

    return summary


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

