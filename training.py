"""
Training utilities for the time-series RNN experiments.
"""

import math
import multiprocessing as mp
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from rnns import build_rnn

# Data containers


class ExpandingFold:
    """Index boundaries for one expanding-window cross-validation fold."""
    def __init__(self, fold, train_start, train_end, val_start, val_end):
        self.fold = fold
        self.train_start = train_start
        self.train_end = train_end
        self.val_start = val_start
        self.val_end = val_end


class PreparedFold:
    """A cross-validation fold after fitting its training-only scaler."""
    def __init__(self, fold, scaler, scaled_values, raw_values):
        self.fold = fold
        self.scaler = scaler
        self.scaled_values = scaled_values
        self.raw_values = raw_values


class PreparedCV:
    """Prepared expanding-window CV data for one dataset."""
    def __init__(self, dataset_name, raw_values, folds):
        self.dataset_name = dataset_name
        self.raw_values = raw_values
        self.folds = folds


class PreparedFinal:
    """Development/test data prepared for the final evaluation stage."""
    def __init__(self, dataset_name, development_values, test_values, combined_raw, combined_scaled, scaler, development_end):
        self.dataset_name = dataset_name
        self.development_values = development_values
        self.test_values = test_values
        self.combined_raw = combined_raw
        self.combined_scaled = combined_scaled
        self.scaler = scaler
        self.development_end = development_end


class WindowedTimeSeriesDataset(Dataset):
    """
    Lazy PyTorch dataset for one-step-ahead time-series windows.
    """
    def __init__(self, scaled_values, target_indices, window_size):
        self.values = np.asarray(scaled_values, dtype=np.float32).reshape(-1)
        self.target_indices = np.asarray(target_indices, dtype=np.int64)
        self.window_size = int(window_size)

    def __len__(self):
        return len(self.target_indices)

    def __getitem__(self, index):
        target_index = int(self.target_indices[index])
        x = torch.from_numpy(
            self.values[target_index - self.window_size : target_index]
        ).unsqueeze(-1)
        y = torch.tensor([self.values[target_index]], dtype=torch.float32)

        return x, y


# Reproducibility and general helpers


def set_seed(seed, deterministic=True):
    """Seed Python, NumPy, and PyTorch for reproducible model initialization.

    The seed must be set before model construction because rnns.py uses Xavier
    initialization, which draws random initial weights.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)

        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def resolve_device(device=None):
    """Return the requested device, or choose CUDA automatically if available."""

    if device is not None:
        return torch.device(device)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    """Count trainable parameters in a model."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _timestamp():
    """Return a compact timestamp for progress messages and result rows."""

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _safe_float(value):
    """Convert a float into a filename-safe representation."""

    return (f"{value:.12g}".replace("+", "").replace("-", "m").replace(".", "p"))


# Expanding-window cross-validation preparation


def make_expanding_folds(n_rows, n_folds=4, initial_train_fraction=0.40, validation_fraction=0.15):
    """
    Construct chronological expanding-window CV fold boundaries.
    """

    total_fraction = initial_train_fraction + n_folds * validation_fraction

    initial_train_end = int(math.floor(n_rows * initial_train_fraction))
    folds = []

    for fold_index in range(n_folds):
        train_end = initial_train_end + int(math.floor(n_rows * validation_fraction * fold_index))
        val_start = train_end
        uses_remaining_data = (fold_index == n_folds - 1 and abs(total_fraction - 1.0) < 1e-9)

        if uses_remaining_data:
            val_end = n_rows
        else:
            val_end = initial_train_end + int(
                math.floor(n_rows * validation_fraction * (fold_index + 1))
            )


        folds.append(
            ExpandingFold(
                fold=fold_index + 1,
                train_start=0,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
            )
        )

    return folds


def prepare_cv_data(
    dataset_name,
    values,
    n_folds=4,
    initial_train_fraction=0.40,
    validation_fraction=0.15,
):
    """
    Prepare all CV folds and fit one scaler per fold.
    """

    raw_values = np.asarray(values, dtype=np.float64).reshape(-1)

    fold_definitions = make_expanding_folds(
        n_rows=len(raw_values),
        n_folds=n_folds,
        initial_train_fraction=initial_train_fraction,
        validation_fraction=validation_fraction,
    )
    prepared_folds = []

    for fold in fold_definitions:
        training_values = raw_values[fold.train_start : fold.train_end]
        scaler = StandardScaler()
        scaler.fit(training_values.reshape(-1, 1))

        # Scale only the chronological region needed by this fold.
        values_needed_for_fold = raw_values[: fold.val_end]
        scaled_values = scaler.transform(
            values_needed_for_fold.reshape(-1, 1)
        ).reshape(-1).astype(np.float32)
        prepared_folds.append(
            PreparedFold(
                fold=fold,
                scaler=scaler,
                scaled_values=scaled_values,
                raw_values=values_needed_for_fold.copy(),
            )
        )

    return PreparedCV(dataset_name=dataset_name, raw_values=raw_values, folds=prepared_folds)


def make_fold_loaders(
    prepared_fold,
    window_size,
    batch_size=128,
    shuffle_train=False,
    num_workers=0,
    pin_memory=False,
):
    """
    Create train and validation DataLoaders for one prepared fold.
    """

    fold = prepared_fold.fold
    train_length = fold.train_end - fold.train_start

    train_targets = np.arange(
        max(fold.train_start + window_size, window_size),
        fold.train_end,
        dtype=np.int64,
    )

    # Validation begins exactly at val_start. The first validation input may
    # therefore use the final window_size observations from the training block.
    validation_targets = np.arange(fold.val_start, fold.val_end, dtype=np.int64)
    train_dataset = WindowedTimeSeriesDataset(
        prepared_fold.scaled_values,
        train_targets,
        window_size,
    )
    validation_dataset = WindowedTimeSeriesDataset(
        prepared_fold.scaled_values,
        validation_targets,
        window_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, validation_loader


# Metrics and model evaluation


def _collect_predictions(model, loader, device):
    """
    Collect predictions, targets, scaled MSE, and model inference time.
    """

    model.eval()
    predictions = []
    targets = []
    squared_error = nn.MSELoss(reduction="sum")
    total_squared_error = 0.0
    total_targets = 0
    inference_seconds = 0.0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            inference_start = time.perf_counter()
            prediction = model(x_batch)

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            inference_seconds += time.perf_counter() - inference_start
            total_squared_error += squared_error(prediction, y_batch).item()
            total_targets += y_batch.numel()
            predictions.append(prediction.detach().cpu().numpy().reshape(-1))
            targets.append(y_batch.detach().cpu().numpy().reshape(-1))

    return (
        np.concatenate(predictions),
        np.concatenate(targets),
        total_squared_error / total_targets,
        inference_seconds,
    )


def _inverse_1d(scaler, values):
    """Convert a one-dimensional standardized series back to original units."""

    return scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).reshape(-1)


def regression_metrics(y_true, y_pred, mase_scale=None):
    """Calculate a broad set of regression and forecasting metrics.

    Returns MSE, RMSE, MAE, optional MASE, Pearson correlation, R-squared,
    mean signed error (forecast bias), and median absolute error.
    """

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    error = y_pred - y_true
    absolute_error = np.abs(error)
    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(absolute_error))
    mean_error = float(np.mean(error))
    median_absolute_error = float(np.median(absolute_error))
    target_variance_sum = float(np.sum((y_true - np.mean(y_true)) ** 2))

    if target_variance_sum > 0:
        r2 = float(1.0 - np.sum(error ** 2) / target_variance_sum)
    else:
        r2 = float("nan")

    true_std = float(np.std(y_true))
    pred_std = float(np.std(y_pred))

    if true_std > 0 and pred_std > 0:
        pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson_r = float("nan")

    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "pearson_r": pearson_r,
        "r2": r2,
        "mean_error": mean_error,
        "median_absolute_error": median_absolute_error,
    }

    if mase_scale is not None:
        metrics["mase"] = (float(mae / mase_scale) if mase_scale > 0 else float("nan"))

    return metrics


def mase_scale_from_training(raw_train, lag=1):
    """
    Calculate the in-sample naive MAE used as the MASE denominator.
    """

    raw_train = np.asarray(raw_train, dtype=np.float64)
    return float(np.mean(np.abs(raw_train[lag:] - raw_train[:-lag])))


def evaluate_model(model, loader, scaler, device, mase_scale=None):
    """Evaluate a model and report metrics in the original data units."""

    (pred_scaled, true_scaled, scaled_mse, inference_seconds) = _collect_predictions(model, loader, device)
    predictions = _inverse_1d(scaler, pred_scaled)
    targets = _inverse_1d(scaler, true_scaled)
    metrics = regression_metrics(targets, predictions, mase_scale)
    metrics["scaled_mse"] = float(scaled_mse)
    metrics["inference_seconds"] = float(inference_seconds)
    metrics["inference_ms_per_sample"] = float(1000.0 * inference_seconds / len(targets))

    return metrics


# Run identifiers, result logging, and checkpoints


def make_run_id(
    dataset,
    architecture,
    window_size,
    hidden_size,
    learning_rate,
    weight_decay,
    fold,
    seed,
):
    """Create a deterministic identifier for one fold-level training run."""

    return (
        f"{dataset}_{architecture}"
        f"_w{window_size}"
        f"_h{hidden_size}"
        f"_lr{_safe_float(learning_rate)}"
        f"_wd{_safe_float(weight_decay)}"
        f"_fold{fold}"
        f"_seed{seed}"
    )


def is_run_completed(results_csv, run_id):
    """Return True when a run already exists with status='completed'."""

    path = Path(results_csv)

    if not path.exists():
        return False

    try:
        results = pd.read_csv(path, usecols=["run_id", "status"])
    except pd.errors.EmptyDataError:
        return False

    matching_rows = results[results["run_id"].astype(str) == str(run_id)]

    return bool((matching_rows["status"].astype(str) == "completed").any())


def write_result_atomic(results_csv, row):
    """
    Insert or replace one result row using an atomic file replacement.
    """

    path = Path(results_csv)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        try:
            results = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            results = pd.DataFrame()
    else:
        results = pd.DataFrame()

    run_id = str(row.get("run_id", ""))

    if (
        not results.empty
        and "run_id" in results.columns
        and run_id
        in results["run_id"].astype(str).values
    ):
        results = results[results["run_id"].astype(str) != run_id]

    results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    results.to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)


def save_checkpoint_atomic(path, payload):
    """Save a checkpoint through a temporary file before replacing the target."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def _checkpoint_payload(
    model,
    architecture,
    input_size,
    hidden_size,
    output_size,
    activation,
    window_size,
    seed,
    best_epoch,
    best_val_loss,
    scaler,
    fold=None,
):
    """Construct the metadata stored with a model checkpoint."""

    payload = {
        "model_state_dict": model.state_dict(),
        "architecture": architecture,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "activation": activation,
        "window_size": window_size,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "scaler_mean": scaler.mean_.copy(),
        "scaler_scale": scaler.scale_.copy(),
        "scaler_var": scaler.var_.copy(),
    }

    if fold is not None:
        payload["fold"] = vars(fold).copy()

    return payload


def save_history_atomic(path, history):
    """Save per-epoch training history using an atomic file replacement."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(history).to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)


def save_predictions_atomic(path, arrays):
    """Save prediction arrays in a compressed NumPy archive atomically."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with open(temporary_path, "wb") as file:
        np.savez_compressed(file, **arrays)

    os.replace(temporary_path, path)


def save_run_predictions(model, train_loader, val_loader, scaler, raw_values, device, path):
    """
    Save train/validation targets and predictions in original units.
    """

    (train_pred_scaled, train_true_scaled, _, _) = _collect_predictions(model, train_loader, device)
    (val_pred_scaled, val_true_scaled, _, _) = _collect_predictions(model, val_loader, device)
    train_indices = np.asarray(train_loader.dataset.target_indices, dtype=np.int64)
    val_indices = np.asarray(val_loader.dataset.target_indices, dtype=np.int64)
    raw_values = np.asarray(raw_values, dtype=np.float64)

    save_predictions_atomic(
        path,
        {
            "train_target_indices": train_indices,
            "train_y_true": _inverse_1d(scaler, train_true_scaled),
            "train_y_pred": _inverse_1d(scaler, train_pred_scaled),
            "train_previous_actual": raw_values[train_indices - 1],
            "val_target_indices": val_indices,
            "val_y_true": _inverse_1d(scaler, val_true_scaled),
            "val_y_pred": _inverse_1d(scaler, val_pred_scaled),
            "val_previous_actual": raw_values[val_indices - 1],
        },
    )


# RNN training


def _train_epoch(model, train_loader, optimizer, loss_function, resolved_device, gradient_clip):
    """Run one training epoch and return the scaled mean squared error."""
    model.train()
    training_loss_sum = 0.0
    training_target_count = 0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(resolved_device)
        y_batch = y_batch.to(resolved_device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x_batch)
        loss = loss_function(prediction, y_batch)

        loss.backward()

        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)

        optimizer.step()
        training_loss_sum += (loss.item() * y_batch.numel())
        training_target_count += (y_batch.numel())

    train_scaled_mse = (training_loss_sum / training_target_count)
    return train_scaled_mse


def train_one_model(
    train_loader,
    val_loader,
    scaler,
    architecture,
    hidden_size,
    learning_rate,
    weight_decay,
    window_size,
    seed=42,
    activation="tanh",
    input_size=1,
    output_size=1,
    max_epochs=100,
    patience=10,
    min_delta=1e-4,
    gradient_clip=1.0,
    device=None,
    checkpoint_path=None,
    fold=None,
    mase_scale=None,
    verbose=True,
    epoch_log_interval=1,
    history_path=None,
):
    """
    Train one RNN on one CV fold with validation-based early stopping.
    """

    set_seed(seed)
    resolved_device = resolve_device(device)
    model = build_rnn(
        architecture,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        activation=activation,
    ).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = nn.MSELoss()

    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_state = None
    history = []
    start_time = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        train_scaled_mse = _train_epoch(
            model, train_loader, optimizer, loss_function, resolved_device, gradient_clip
        )
        _, _, val_scaled_mse, _ = _collect_predictions(model, val_loader, resolved_device)
        improved = (val_scaled_mse < best_validation_loss - min_delta)

        if improved:
            best_validation_loss = val_scaled_mse
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter
                in model.state_dict().items()
            }

        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train_scaled_mse": train_scaled_mse,
                "val_scaled_mse": val_scaled_mse,
                "best_val_scaled_mse": best_validation_loss,
                "improved": improved,
                "epochs_without_improvement": (epochs_without_improvement),
            }
        )

        if history_path is not None:
            save_history_atomic(history_path, history)

        if verbose and (
            epoch == 1
            or epoch % epoch_log_interval == 0
            or improved
            or epochs_without_improvement >= patience
        ):
            print(
                f"[{_timestamp()}] "
                f"epoch {epoch}/{max_epochs} | "
                f"train_scaled_mse={train_scaled_mse:.6g} | "
                f"val_scaled_mse={val_scaled_mse:.6g} | "
                f"best={best_validation_loss:.6g} | "
                f"patience="
                f"{epochs_without_improvement}/{patience}"
            )

        if epochs_without_improvement >= patience:
            break

    if best_state is None:
        best_state = {
            name: parameter.detach().cpu().clone()
            for name, parameter
            in model.state_dict().items()
        }
        best_epoch = epoch
        best_validation_loss = (val_scaled_mse)

    # Restore the weights from the best validation epoch before evaluation.
    model.load_state_dict(best_state)
    model.to(resolved_device)
    train_metrics = evaluate_model(model, train_loader, scaler, resolved_device, mase_scale)
    validation_metrics = evaluate_model(model, val_loader, scaler, resolved_device, mase_scale)
    runtime_seconds = (time.perf_counter() - start_time)
    metrics = {
        "epochs_trained": epoch,
        "best_epoch": best_epoch,
        "best_val_scaled_mse": best_validation_loss,
        "train_mse": train_metrics["mse"],
        "train_rmse": train_metrics["rmse"],
        "train_mae": train_metrics["mae"],
        "train_pearson_r": train_metrics["pearson_r"],
        "train_r2": train_metrics["r2"],
        "train_mean_error": train_metrics["mean_error"],
        "train_median_absolute_error": (train_metrics["median_absolute_error"]),
        "val_mse": validation_metrics["mse"],
        "val_rmse": validation_metrics["rmse"],
        "val_mae": validation_metrics["mae"],
        "val_pearson_r": validation_metrics["pearson_r"],
        "val_r2": validation_metrics["r2"],
        "val_mean_error": validation_metrics["mean_error"],
        "val_median_absolute_error": (validation_metrics["median_absolute_error"]),
        "inference_seconds": validation_metrics["inference_seconds"],
        "inference_ms_per_sample": validation_metrics["inference_ms_per_sample"],
        "generalization_gap": (validation_metrics["rmse"] - train_metrics["rmse"]),
        "runtime_seconds": runtime_seconds,
        "n_parameters": count_parameters(model),
    }

    if "mase" in train_metrics:
        metrics["train_mase"] = (train_metrics["mase"])
        metrics["val_mase"] = (validation_metrics["mase"])

    if checkpoint_path is not None:
        checkpoint = _checkpoint_payload(
            model=model,
            architecture=architecture,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            activation=activation,
            window_size=window_size,
            seed=seed,
            best_epoch=best_epoch,
            best_val_loss=best_validation_loss,
            scaler=scaler,
            fold=fold,
        )
        save_checkpoint_atomic(checkpoint_path, checkpoint)

    return model, metrics


# Persistence baseline


def evaluate_persistence_fold(prepared_fold, mase_lag=1):
    """Evaluate y_hat(t) = y(t-1) on one validation fold."""

    fold = prepared_fold.fold
    raw_values = prepared_fold.raw_values
    true_values = raw_values[fold.val_start : fold.val_end]
    predictions = raw_values[fold.val_start - 1 : fold.val_end - 1]
    mase_scale = mase_scale_from_training(raw_values[fold.train_start : fold.train_end], mase_lag)

    return regression_metrics(true_values, predictions, mase_scale)


def run_persistence_baseline(prepared_cv, results_csv=None, mase_lag=1, verbose=True):
    """Evaluate the persistence baseline on every prepared CV fold."""

    rows = []

    for prepared_fold in prepared_cv.folds:
        fold = prepared_fold.fold
        run_id = (f"{prepared_cv.dataset_name}" f"_persistence_fold{fold.fold}")

        if (results_csv is not None and is_run_completed(results_csv, run_id)):
            if verbose:
                print(f"[{_timestamp()}] " f"SKIP {run_id}")

            continue

        metrics = evaluate_persistence_fold(prepared_fold, mase_lag)
        row = {
            "run_id": run_id,
            "dataset": prepared_cv.dataset_name,
            "architecture": "persistence",
            "fold": fold.fold,
            "seed": np.nan,
            "window_size": 1,
            "hidden_size": np.nan,
            "learning_rate": np.nan,
            "weight_decay": np.nan,
            "batch_size": np.nan,
            "max_epochs": np.nan,
            "epochs_trained": 0,
            "best_epoch": 0,
            "train_mse": np.nan,
            "train_rmse": np.nan,
            "train_mae": np.nan,
            "train_mase": np.nan,
            "val_mse": metrics["mse"],
            "val_rmse": metrics["rmse"],
            "val_mae": metrics["mae"],
            "val_mase": metrics.get("mase", np.nan),
            "val_pearson_r": metrics["pearson_r"],
            "val_r2": metrics["r2"],
            "val_mean_error": metrics["mean_error"],
            "val_median_absolute_error": (metrics["median_absolute_error"]),
            "generalization_gap": np.nan,
            "runtime_seconds": 0.0,
            "n_parameters": 0,
            "checkpoint_path": "",
            "status": "completed",
            "timestamp": _timestamp(),
        }

        if results_csv is not None:
            write_result_atomic(results_csv, row)

        rows.append(row)

        if verbose:
            print(f"[{_timestamp()}] " f"DONE {run_id} | " f"val_rmse=" f"{metrics['rmse']:.6g}")

    return pd.DataFrame(rows)


# Cross-validation orchestration


def run_cv_configuration(
    prepared_cv,
    architecture,
    window_size,
    hidden_size,
    learning_rate,
    weight_decay,
    batch_size=128,
    seed=42,
    activation="tanh",
    max_epochs=100,
    patience=10,
    min_delta=1e-4,
    gradient_clip=1.0,
    results_csv=None,
    checkpoint_dir=None,
    prediction_dir=None,
    history_dir=None,
    device=None,
    shuffle_train=False,
    num_workers=0,
    pin_memory=False,
    mase_lag=1,
    verbose=True,
    epoch_log_interval=1,
    continue_on_error=False,
):
    """
    Train one hyperparameter configuration across every CV fold.
    """

    rows = []

    for prepared_fold in prepared_cv.folds:
        fold = prepared_fold.fold
        run_id = make_run_id(
            dataset=prepared_cv.dataset_name,
            architecture=architecture,
            window_size=window_size,
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            fold=fold.fold,
            seed=seed,
        )

        if (results_csv is not None and is_run_completed(results_csv, run_id)):
            if verbose:
                print(f"[{_timestamp()}] " f"SKIP {run_id}")

            continue

        checkpoint_path = None
        prediction_path = None
        history_path = None

        if checkpoint_dir is not None:
            checkpoint_path = (Path(checkpoint_dir) / prepared_cv.dataset_name / f"{run_id}.pt")

        if prediction_dir is not None:
            prediction_path = (Path(prediction_dir) / prepared_cv.dataset_name / f"{run_id}.npz")

        if history_dir is not None:
            history_path = (Path(history_dir) / prepared_cv.dataset_name / f"{run_id}.csv")

        if verbose:
            print(f"[{_timestamp()}] " f"START {run_id}")

        try:
            train_loader, val_loader = make_fold_loaders(
                prepared_fold=prepared_fold,
                window_size=window_size,
                batch_size=batch_size,
                shuffle_train=shuffle_train,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            raw_training_values = (prepared_fold.raw_values[fold.train_start : fold.train_end])
            mase_scale = mase_scale_from_training(raw_training_values, mase_lag)
            model, metrics = train_one_model(
                train_loader=train_loader,
                val_loader=val_loader,
                scaler=prepared_fold.scaler,
                architecture=architecture,
                hidden_size=hidden_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                window_size=window_size,
                seed=seed,
                activation=activation,
                max_epochs=max_epochs,
                patience=patience,
                min_delta=min_delta,
                gradient_clip=gradient_clip,
                device=device,
                checkpoint_path=checkpoint_path,
                fold=fold,
                mase_scale=mase_scale,
                verbose=verbose,
                epoch_log_interval=epoch_log_interval,
                history_path=history_path,
            )

            if prediction_path is not None:
                save_run_predictions(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    scaler=prepared_fold.scaler,
                    raw_values=prepared_fold.raw_values,
                    device=resolve_device(device),
                    path=prediction_path,
                )

            row = {
                "run_id": run_id,
                "dataset": prepared_cv.dataset_name,
                "architecture": architecture,
                "fold": fold.fold,
                "seed": seed,
                "window_size": window_size,
                "hidden_size": hidden_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "max_epochs": max_epochs,
                **metrics,
                "checkpoint_path": (str(checkpoint_path) if checkpoint_path is not None else ""),
                "prediction_path": (str(prediction_path) if prediction_path is not None else ""),
                "history_path": (str(history_path) if history_path is not None else ""),
                "status": "completed",
                "timestamp": _timestamp(),
            }

            if results_csv is not None:
                write_result_atomic(results_csv, row)

            rows.append(row)

            if verbose:
                print(
                    f"[{_timestamp()}] "
                    f"DONE {run_id} | "
                    f"val_rmse="
                    f"{metrics['val_rmse']:.6g} | "
                    f"best_epoch="
                    f"{metrics['best_epoch']}"
                )

        except Exception as exception:
            failed_row = {
                "run_id": run_id,
                "dataset": prepared_cv.dataset_name,
                "architecture": architecture,
                "fold": fold.fold,
                "seed": seed,
                "window_size": window_size,
                "hidden_size": hidden_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "max_epochs": max_epochs,
                "checkpoint_path": (str(checkpoint_path) if checkpoint_path is not None else ""),
                "prediction_path": (str(prediction_path) if prediction_path is not None else ""),
                "history_path": (str(history_path) if history_path is not None else ""),
                "status": "failed",
                "error": repr(exception),
                "timestamp": _timestamp(),
            }

            if results_csv is not None:
                write_result_atomic(results_csv, failed_row)

            if verbose:
                print(f"[{_timestamp()}] " f"FAILED {run_id}: " f"{exception}")

            if not continue_on_error:
                raise

    return pd.DataFrame(rows)


def _run_single_cv_job(job):
    """
    Train one fold-level CV job inside a worker process.
    """

    torch_threads = int(job["torch_threads"])


    torch.set_num_threads(torch_threads)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch allows this setting only before parallel work has started.
        # A worker that has already initialized inter-op threads can continue
        # safely without changing the value.
        pass

    prepared_fold = job["prepared_fold"]
    fold = prepared_fold.fold
    architecture = job["architecture"]
    window_size = int(job["window_size"])
    hidden_size = int(job["hidden_size"])
    learning_rate = float(job["learning_rate"])
    weight_decay = float(job["weight_decay"])
    batch_size = int(job["batch_size"])
    seed = int(job["seed"])
    activation = job["activation"]
    max_epochs = int(job["max_epochs"])
    patience = int(job["patience"])
    min_delta = float(job["min_delta"])
    gradient_clip = job["gradient_clip"]
    device = job["device"]
    shuffle_train = bool(job["shuffle_train"])
    num_workers = int(job["num_workers"])
    pin_memory = bool(job["pin_memory"])
    mase_lag = int(job["mase_lag"])
    verbose = bool(job["verbose"])
    epoch_log_interval = int(job["epoch_log_interval"])
    run_id = job["run_id"]
    dataset_name = job["dataset_name"]
    checkpoint_path = job["checkpoint_path"]
    prediction_path = job["prediction_path"]
    history_path = job["history_path"]
    started = time.perf_counter()

    if verbose:
        print(f"[{_timestamp()}] " f"WORKER START {run_id}", flush=True)

    try:
        train_loader, val_loader = make_fold_loaders(
            prepared_fold=prepared_fold,
            window_size=window_size,
            batch_size=batch_size,
            shuffle_train=shuffle_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        raw_training_values = prepared_fold.raw_values[fold.train_start : fold.train_end]
        mase_scale = mase_scale_from_training(raw_training_values, mase_lag)
        model, metrics = train_one_model(
            train_loader=train_loader,
            val_loader=val_loader,
            scaler=prepared_fold.scaler,
            architecture=architecture,
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            window_size=window_size,
            seed=seed,
            activation=activation,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
            gradient_clip=gradient_clip,
            device=device,
            checkpoint_path=checkpoint_path,
            fold=fold,
            mase_scale=mase_scale,
            verbose=verbose,
            epoch_log_interval=epoch_log_interval,
            history_path=history_path,
        )

        if prediction_path is not None:
            save_run_predictions(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                scaler=prepared_fold.scaler,
                raw_values=prepared_fold.raw_values,
                device=resolve_device(device),
                path=prediction_path,
            )

        row = {
            "run_id": run_id,
            "dataset": dataset_name,
            "architecture": architecture,
            "fold": fold.fold,
            "seed": seed,
            "window_size": window_size,
            "hidden_size": hidden_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            **metrics,
            "checkpoint_path": (str(checkpoint_path) if checkpoint_path is not None else ""),
            "prediction_path": (str(prediction_path) if prediction_path is not None else ""),
            "history_path": (str(history_path) if history_path is not None else ""),
            "status": "completed",
            "timestamp": _timestamp(),
            "worker_runtime_seconds": (time.perf_counter() - started),
        }

        if verbose:
            print(
                f"[{_timestamp()}] "
                f"WORKER DONE {run_id} | "
                f"val_rmse={metrics['val_rmse']:.6g}",
                flush=True,
            )

        return row

    except Exception as exception:
        return {
            "run_id": run_id,
            "dataset": dataset_name,
            "architecture": architecture,
            "fold": fold.fold,
            "seed": seed,
            "window_size": window_size,
            "hidden_size": hidden_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "checkpoint_path": (str(checkpoint_path) if checkpoint_path is not None else ""),
            "prediction_path": (str(prediction_path) if prediction_path is not None else ""),
            "history_path": (str(history_path) if history_path is not None else ""),
            "status": "failed",
            "error": repr(exception),
            "timestamp": _timestamp(),
            "worker_runtime_seconds": (time.perf_counter() - started),
        }


def run_cv_configurations_parallel(
    prepared_cv,
    configurations,
    n_jobs=4,
    batch_size=128,
    seed=42,
    activation="tanh",
    max_epochs=100,
    patience=10,
    min_delta=1e-4,
    gradient_clip=1.0,
    results_csv=None,
    checkpoint_dir=None,
    prediction_dir=None,
    history_dir=None,
    device="cpu",
    shuffle_train=False,
    num_workers=0,
    pin_memory=False,
    mase_lag=1,
    worker_torch_threads=1,
    verbose=True,
    worker_verbose=False,
    epoch_log_interval=1,
    continue_on_error=False,
):
    """Train multiple CV configurations concurrently using worker processes.

    Each configuration dictionary must contain:

        architecture
        window_size
        hidden_size
        learning_rate
        weight_decay

    Optional per-configuration overrides are supported for:

        batch_size
        seed
        activation
        max_epochs
        patience
        min_delta
        gradient_clip

    Parallelism is performed at the fold/configuration level. For example,
    with four folds and ``n_jobs=4``, four independent fold-level models can
    train at the same time.

    Shared results are safe because workers never write ``results_csv``.
    Workers write only unique run-specific artifacts. The parent process writes
    one CSV row after each worker completes.

    This function is intended primarily for CPU training. Running several
    models simultaneously on one GPU usually causes memory/compute contention,
    so ``n_jobs > 1`` is rejected for CUDA devices.

    Args:
        prepared_cv:
            Dataset prepared once with ``prepare_cv_data``.
        configurations:
            Sequence of hyperparameter dictionaries.
        n_jobs:
            Number of worker processes. Start with 4 on a 12-thread CPU.
        worker_torch_threads:
            Number of PyTorch intra-op CPU threads available to each worker.
            A value of 1 avoids oversubscribing the CPU when several workers
            run concurrently.
        worker_verbose:
            If True, workers print epoch-level progress. False is recommended
            because output from several processes can become interleaved.

    Returns:
        DataFrame containing rows completed during this call. Runs already
        marked completed in ``results_csv`` are skipped.
    """


    resolved_device = resolve_device(device)

    if resolved_device.type == "cuda" and n_jobs > 1:
        raise ValueError(
            "Parallel CV with n_jobs > 1 is intended for CPU training. "
            "Use n_jobs=1 for a single GPU."
        )

    jobs = []
    skipped = 0

    for configuration in configurations:

        architecture = str(configuration["architecture"])
        window_size = int(configuration["window_size"])
        hidden_size = int(configuration["hidden_size"])
        learning_rate = float(configuration["learning_rate"])
        weight_decay = float(configuration["weight_decay"])

        configuration_batch_size = int(configuration.get("batch_size", batch_size))
        configuration_seed = int(configuration.get("seed", seed))
        configuration_activation = str(configuration.get("activation", activation))
        configuration_max_epochs = int(configuration.get("max_epochs", max_epochs))
        configuration_patience = int(configuration.get("patience", patience))

        configuration_min_delta = float(configuration.get("min_delta", min_delta))
        configuration_gradient_clip = (configuration.get("gradient_clip", gradient_clip))

        for prepared_fold in prepared_cv.folds:
            fold = prepared_fold.fold
            run_id = make_run_id(
                dataset=prepared_cv.dataset_name,
                architecture=architecture,
                window_size=window_size,
                hidden_size=hidden_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                fold=fold.fold,
                seed=configuration_seed,
            )

            if (results_csv is not None and is_run_completed(results_csv, run_id)):
                skipped += 1

                if verbose:
                    print(f"[{_timestamp()}] " f"SKIP {run_id}")

                continue

            checkpoint_path = None
            prediction_path = None
            history_path = None

            if checkpoint_dir is not None:
                checkpoint_path = (Path(checkpoint_dir) / prepared_cv.dataset_name / f"{run_id}.pt")

            if prediction_dir is not None:
                prediction_path = (
                    Path(prediction_dir)
                    / prepared_cv.dataset_name
                    / f"{run_id}.npz"
                )

            if history_dir is not None:
                history_path = (Path(history_dir) / prepared_cv.dataset_name / f"{run_id}.csv")

            jobs.append(
                {
                    "prepared_fold": prepared_fold,
                    "dataset_name": (prepared_cv.dataset_name),
                    "architecture": architecture,
                    "window_size": window_size,
                    "hidden_size": hidden_size,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "batch_size": (configuration_batch_size),
                    "seed": configuration_seed,
                    "activation": (configuration_activation),
                    "max_epochs": (configuration_max_epochs),
                    "patience": (configuration_patience),
                    "min_delta": (configuration_min_delta),
                    "gradient_clip": (configuration_gradient_clip),
                    "device": str(resolved_device),
                    "shuffle_train": shuffle_train,
                    "num_workers": num_workers,
                    "pin_memory": pin_memory,
                    "mase_lag": mase_lag,
                    "verbose": worker_verbose,
                    "epoch_log_interval": (epoch_log_interval),
                    "torch_threads": (worker_torch_threads),
                    "run_id": run_id,
                    "checkpoint_path": (checkpoint_path),
                    "prediction_path": (prediction_path),
                    "history_path": (history_path),
                }
            )

    if verbose:
        print(
            f"[{_timestamp()}] "
            f"parallel CV: {len(jobs)} jobs queued, "
            f"{skipped} already completed, "
            f"{n_jobs} workers"
        )

    if len(jobs) == 0:
        return pd.DataFrame()

    completed_rows = []
    started = time.perf_counter()

    # Spawn is safer than fork for PyTorch and is reliable when the worker
    # function lives in this importable module rather than in a notebook cell.
    context = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=context) as executor:
        future_to_run_id = {
            executor.submit(_run_single_cv_job, job): job["run_id"]
            for job in jobs
        }
        total_jobs = len(future_to_run_id)
        completed_count = 0

        for future in as_completed(future_to_run_id):
            run_id = future_to_run_id[future]
            row = future.result()
            completed_count += 1
            completed_rows.append(row)

            if results_csv is not None:
                write_result_atomic(results_csv, row)

            if verbose:
                elapsed = (time.perf_counter() - started)
                average_job_time = (elapsed / completed_count)
                remaining_jobs = (total_jobs - completed_count)

                # Because several workers run simultaneously, this is a rough
                # queue-level ETA rather than a strict sum of job runtimes.
                remaining_batches = (remaining_jobs / max(1, n_jobs))
                eta_seconds = (average_job_time * remaining_batches)
                status = row.get("status", "unknown")
                print(
                    f"[{_timestamp()}] "
                    f"{completed_count}/{total_jobs} "
                    f"{status.upper()} {run_id} | "
                    f"elapsed={elapsed / 60:.1f} min | "
                    f"rough ETA={eta_seconds / 60:.1f} min"
                )

            if (row.get("status") == "failed" and not continue_on_error):
                for pending_future in future_to_run_id:
                    pending_future.cancel()

                raise RuntimeError(
                    f"Parallel run failed: {run_id}: "
                    f"{row.get('error', 'unknown error')}"
                )

    return pd.DataFrame(completed_rows)


def summarize_cv(results):
    """Summarize validation performance across completed folds."""

    if results.empty:
        return {}

    if "status" in results.columns:
        completed = results[results["status"] == "completed"]
    else:
        completed = results

    summary = {}
    metrics = [
        "val_mse",
        "val_rmse",
        "val_mae",
        "val_mase",
        "val_pearson_r",
        "val_r2",
        "val_mean_error",
        "val_median_absolute_error",
        "generalization_gap",
        "runtime_seconds",
        "inference_seconds",
        "inference_ms_per_sample",
    ]

    for metric in metrics:
        if metric not in completed.columns:
            continue

        values = pd.to_numeric(completed[metric], errors="coerce").dropna()

        if len(values) == 0:
            continue

        summary[f"mean_{metric}"] = float(values.mean())
        summary[f"std_{metric}"] = (float(values.std(ddof=1)) if len(values) > 1 else 0.0)

    return summary


# Final development/test training


def prepare_final_data(dataset_name, development_values, test_values):
    """Fit one scaler on the full development set for final evaluation.

    The test set is never used when fitting this scaler. Development and test
    values are then concatenated only so the first test prediction can use the
    final development observations as its historical input window.
    """

    development = np.asarray(development_values, dtype=np.float64).reshape(-1)
    test = np.asarray(test_values, dtype=np.float64).reshape(-1)

    if (not np.isfinite(development).all() or not np.isfinite(test).all()):
        raise ValueError("development or test values contain NaN or infinite values")

    scaler = StandardScaler()
    scaler.fit(development.reshape(-1, 1))
    combined_raw = np.concatenate([development, test])
    combined_scaled = scaler.transform(combined_raw.reshape(-1, 1)).reshape(-1).astype(np.float32)

    return PreparedFinal(
        dataset_name=dataset_name,
        development_values=development,
        test_values=test,
        combined_raw=combined_raw,
        combined_scaled=combined_scaled,
        scaler=scaler,
        development_end=len(development),
    )


def make_final_loaders(
    prepared,
    window_size,
    batch_size=128,
    shuffle_train=False,
    num_workers=0,
    pin_memory=False,
):
    """Create full-development training and untouched-test DataLoaders."""

    if window_size >= prepared.development_end:
        raise ValueError("window_size is too large for development data")

    training_targets = np.arange(window_size, prepared.development_end, dtype=np.int64)
    test_targets = np.arange(
        prepared.development_end,
        len(prepared.combined_scaled),
        dtype=np.int64,
    )
    training_dataset = WindowedTimeSeriesDataset(
        prepared.combined_scaled,
        training_targets,
        window_size,
    )
    test_dataset = WindowedTimeSeriesDataset(prepared.combined_scaled, test_targets, window_size)
    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return training_loader, test_loader


def train_final_model(
    prepared,
    architecture,
    window_size,
    hidden_size,
    learning_rate,
    weight_decay,
    epochs,
    batch_size=128,
    seed=42,
    activation="tanh",
    gradient_clip=1.0,
    device=None,
    checkpoint_path=None,
    shuffle_train=False,
    num_workers=0,
    pin_memory=False,
    mase_lag=1,
    verbose=True,
    epoch_log_interval=1,
    prediction_path=None,
    history_path=None,
):
    """Train a selected configuration on all development data and test once.

    This function should only be called after hyperparameter/model selection is
    finished. Unlike CV training, there is no validation-based early stopping
    here because the validation folds have already been used during model
    selection. The notebook should supply a fixed number of epochs chosen from
    the CV results, for example the median best epoch of the selected runs.
    """

    set_seed(seed)
    resolved_device = resolve_device(device)
    train_loader, test_loader = make_final_loaders(
        prepared=prepared,
        window_size=window_size,
        batch_size=batch_size,
        shuffle_train=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    model = build_rnn(
        architecture,
        input_size=1,
        hidden_size=hidden_size,
        output_size=1,
        activation=activation,
    ).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    loss_function = nn.MSELoss()
    start_time = time.perf_counter()
    history = []

    for epoch in range(1, epochs + 1):
        train_scaled_mse = _train_epoch(
            model, train_loader, optimizer, loss_function, resolved_device, gradient_clip
        )
        history.append({ "epoch": epoch, "train_scaled_mse": train_scaled_mse, })

        if history_path is not None:
            save_history_atomic(history_path, history)

        if verbose and (epoch == 1 or epoch % epoch_log_interval == 0 or epoch == epochs):
            print(
                f"[{_timestamp()}] "
                f"final epoch {epoch}/{epochs} | "
                f"train_scaled_mse="
                f"{train_scaled_mse:.6g}"
            )

    mase_scale = mase_scale_from_training(prepared.development_values, mase_lag)
    train_metrics = evaluate_model(
        model,
        train_loader,
        prepared.scaler,
        resolved_device,
        mase_scale,
    )
    test_metrics = evaluate_model(model, test_loader, prepared.scaler, resolved_device, mase_scale)
    metrics = {
        "train_mse": train_metrics["mse"],
        "train_rmse": train_metrics["rmse"],
        "train_mae": train_metrics["mae"],
        "train_mase": train_metrics.get("mase", float("nan")),
        "train_pearson_r": train_metrics["pearson_r"],
        "train_r2": train_metrics["r2"],
        "train_mean_error": train_metrics["mean_error"],
        "train_median_absolute_error": (train_metrics["median_absolute_error"]),
        "test_mse": test_metrics["mse"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_mase": test_metrics.get("mase", float("nan")),
        "test_pearson_r": test_metrics["pearson_r"],
        "test_r2": test_metrics["r2"],
        "test_mean_error": test_metrics["mean_error"],
        "test_median_absolute_error": (test_metrics["median_absolute_error"]),
        "inference_seconds": test_metrics["inference_seconds"],
        "inference_ms_per_sample": test_metrics["inference_ms_per_sample"],
        "runtime_seconds": (time.perf_counter() - start_time),
        "n_parameters": count_parameters(model),
    }

    if prediction_path is not None:
        combined_raw = prepared.combined_raw
        (train_pred_scaled, train_true_scaled, _, _) = _collect_predictions(model, train_loader, resolved_device)
        (test_pred_scaled, test_true_scaled, _, _) = _collect_predictions(model, test_loader, resolved_device)
        train_indices = np.asarray(train_loader.dataset.target_indices, dtype=np.int64)
        test_indices = np.asarray(test_loader.dataset.target_indices, dtype=np.int64)

        save_predictions_atomic(
            prediction_path,
            {
                "train_target_indices": train_indices,
                "train_y_true": _inverse_1d(prepared.scaler, train_true_scaled),
                "train_y_pred": _inverse_1d(prepared.scaler, train_pred_scaled),
                "train_previous_actual": (combined_raw[train_indices - 1]),
                "test_target_indices": test_indices,
                "test_y_true": _inverse_1d(prepared.scaler, test_true_scaled),
                "test_y_pred": _inverse_1d(prepared.scaler, test_pred_scaled),
                "test_previous_actual": (combined_raw[test_indices - 1]),
            },
        )

    if checkpoint_path is not None:
        checkpoint = _checkpoint_payload(
            model=model,
            architecture=architecture,
            input_size=1,
            hidden_size=hidden_size,
            output_size=1,
            activation=activation,
            window_size=window_size,
            seed=seed,
            best_epoch=epochs,
            best_val_loss=float("nan"),
            scaler=prepared.scaler,
            fold=None,
        )
        save_checkpoint_atomic(checkpoint_path, checkpoint)

    return model, metrics
