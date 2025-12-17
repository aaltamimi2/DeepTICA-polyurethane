#!/usr/bin/env python3
"""
Stage 2: Train DeepTICA from Bootstrap OPES Data (Desorption-Focused)
====================================================================

Updates:
- Uses desorption-prioritized descriptors (ORDER MATTERS):
    ['dZ','contacts','rg_lig','ree','asph_lig']
- NO feature scaling applied - raw descriptors are used directly
- Unweighted training (logweights=None) -> uses ALL frames equally
- Much more comprehensive visualization: ~15–20 plots saved to output dir

Usage:
  python submit-deep-tica-UNBIASED-2.py
  python submit-deep-tica-UNBIASED-2.py --sweep
  python submit-deep-tica-UNBIASED-2.py --lr 5e-4 --lag 100 --hidden 128
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

from mlcolvar.cvs import DeepTICA
import lightning as L
from mlcolvar.data import DictModule, DictDataset

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from pathlib import Path
import argparse
import json
from itertools import product
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# ==================== ARGUMENT PARSING ====================
parser = argparse.ArgumentParser(description="Train DeepTICA with optional parameter sweep")

parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")

parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
parser.add_argument("--lag", type=int, default=50, help="Lag time in frames (default: 50)")
parser.add_argument("--hidden", type=int, default=64, help="Hidden layer size (default: 64)")
parser.add_argument("--layers", type=int, default=2, help="Number of hidden layers (default: 2)")
parser.add_argument("--epochs", type=int, default=500, help="Max epochs (default: 500)")
parser.add_argument("--batch", type=int, default=256, help="Batch size (default: 256)")
parser.add_argument("--n_cvs", type=int, default=2, help="Number of CVs to learn (default: 2)")

parser.add_argument("--colvar", type=str, default="COLVAR_BOOTSTRAP", help="COLVAR file path")
parser.add_argument("--output", type=str, default="deeptica_output", help="Output directory")

# Plotting knobs
parser.add_argument("--plot_stride", type=int, default=1, help="Subsample stride for plotting heavy scatter plots (default: 1)")
parser.add_argument("--pairplot_max", type=int, default=15000, help="Max points for pairplot (default: 15000)")
parser.add_argument("--hexbin_gridsize", type=int, default=60, help="Hexbin gridsize (default: 60)")

# --- Desorption-alignment auxiliary loss ---
parser.add_argument("--aux_weight", type=float, default=1.0,
                    help="Weight of auxiliary desorption-correlation loss (default: 1.0).")
parser.add_argument("--use_aux", action="store_true",
                    help="Enable auxiliary loss that aligns CV1 with -dZ and -contacts.")

# --- Blocked k-fold CV ---
parser.add_argument("--kfold", type=int, default=1,
                    help="Number of blocked folds (1 = no CV, just a single train/val).")
parser.add_argument("--purge", action="store_true",
                    help="If set, removes lag_time frames around each validation block from training (recommended).")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility (default: 42).")


args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
L.seed_everything(args.seed, workers=True)
from torch.utils.data import DataLoader

class ExplicitDictDataModule(L.LightningDataModule):
    """
    A LightningDataModule that *guarantees* the exact train/val datasets are used.
    Works with mlcolvar.data.DictDataset (it behaves like a torch Dataset).
    """
    def __init__(self, train_ds, val_ds, batch_size=256, num_workers=0, shuffle=True):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.shuffle = bool(shuffle)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )


# ==================== PRINT HEADER ====================
print("=" * 80)
print("STAGE 2: Training DeepTICA from Bootstrap Data (Desorption-Focused)")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ==================== 1. LOAD BOOTSTRAP DATA ====================
print("\n1. Loading bootstrap trajectory data...")

columns = None
with open(args.colvar, "r") as f:
    for line in f:
        if line.startswith("#! FIELDS"):
            columns = line.split()[2:]
            break

if columns is None:
    raise RuntimeError("Could not find '#! FIELDS' line in COLVAR file to parse column names.")

colvar = pd.read_csv(args.colvar, sep=r"\s+", comment="#", names=columns, header=None)

print(f"   Loaded {len(colvar):,} frames")
print(f"   Columns: {colvar.columns.tolist()}")
print(f"   Time range: {colvar['time'].min():.1f} - {colvar['time'].max():.1f} ps")
print(f"   Duration: {(colvar['time'].max() - colvar['time'].min()) / 1000:.2f} ns")


# ==================== 2. DEFINE DESCRIPTORS ====================
print("\n2. Preparing features for DeepTICA...")

# ORDER MATTERS! This is the trained order and must match PLUMED ARG list later.
descriptor_cols = [
    "dZ",
    "contacts",
    "rg_lig",
    "ree",
    "asph_lig",
]

missing = [c for c in descriptor_cols if c not in colvar.columns]
if missing:
    print(f"   ERROR: Missing required columns: {missing}")
    print("   Fix your PLUMED/COLVAR to include these fields.")
    exit(1)

print(f"   Using descriptors (in order): {descriptor_cols}")

X = colvar[descriptor_cols].values.astype(np.float32)
t = colvar["time"].values.astype(np.float64)

print(f"   Feature matrix shape: {X.shape}")

# Feature statistics
print("\n   Feature statistics:")
feature_stats = {}
for i, col in enumerate(descriptor_cols):
    feature_stats[col] = {
        "mean": float(X[:, i].mean()),
        "std": float(X[:, i].std()),
        "min": float(X[:, i].min()),
        "max": float(X[:, i].max()),
    }
    s = feature_stats[col]
    print(f"   {col:15s}: [{s['min']:.3f}, {s['max']:.3f}] (mean: {s['mean']:.3f}, std: {s['std']:.3f})")

# Keep track of descriptor indices for later use
dZ_idx = descriptor_cols.index("dZ")
contacts_idx = descriptor_cols.index("contacts")


# ==================== 3. (OPTIONAL) OPES BIAS INFO (NOT USED FOR TRAINING) ====================
# We keep a minimal diagnostic print, but training is unweighted (logweights=None).
print("\n3. OPES bias diagnostics (training will be UNWEIGHTED)...")
if "opes.bias" in colvar.columns:
    bias = colvar["opes.bias"].values.astype(np.float64)
    print(f"   opes.bias range: [{bias.min():.2f}, {bias.max():.2f}] kJ/mol")
else:
    print("   Note: 'opes.bias' column not found. Skipping bias diagnostics.")


# ==================== 4. HELPER FUNCTIONS + AUX-LOSS MODEL + BLOCKED KFOLD ====================

def create_timelagged_dataset_masked(X, lag_time, mask_samples=None, logweights=None):
    """
    Creates a time-lagged DictDataset using sample-level masking.
    Sample i corresponds to pair (t0=i, t1=i+lag_time).
    mask_samples is a boolean array of length (n_frames - lag_time).
    """
    n_frames = len(X)
    n_samples = n_frames - lag_time
    if n_samples <= 0:
        raise ValueError("Not enough frames for chosen lag_time.")

    if mask_samples is None:
        mask_samples = np.ones(n_samples, dtype=bool)
    else:
        mask_samples = np.asarray(mask_samples, dtype=bool)
        assert mask_samples.shape[0] == n_samples, "mask_samples must have length n_frames - lag_time"

    idx = np.where(mask_samples)[0]

    X_t0 = torch.FloatTensor(X[idx])
    X_t1 = torch.FloatTensor(X[idx + lag_time])

    data_dict = {"data": X_t0, "data_lag": X_t1}

    # Unweighted training requested -> keep support but default to ones.
    if logweights is not None:
        lw = np.asarray(logweights, dtype=np.float64)
        lw0 = lw[idx]
        lw1 = lw[idx + lag_time]

        w0 = np.exp(lw0 - lw0.max())
        w1 = np.exp(lw1 - lw1.max())

        w0 = w0 / (w0.mean() + 1e-12)
        w1 = w1 / (w1.mean() + 1e-12)

        data_dict["weights"] = torch.FloatTensor(w0.astype(np.float32))
        data_dict["weights_lag"] = torch.FloatTensor(w1.astype(np.float32))
    else:
        data_dict["weights"] = torch.ones(len(idx), dtype=torch.float32)
        data_dict["weights_lag"] = torch.ones(len(idx), dtype=torch.float32)

    return DictDataset(data_dict)


def make_blocked_folds(n_frames, k, lag_time, purge=True):
    """
    Returns list of folds with (train_mask_samples, val_mask_samples).
    Masks are defined at the *sample* level (length n_frames - lag_time).
    Validation blocks are contiguous in time.

    purge=True removes a gap of +/- lag_time around validation block from training
    to reduce leakage from time correlation / shared lag pairs.
    """
    if k < 2:
        return None  # means no CV; use standard random split or a single blocked split outside

    n_samples = n_frames - lag_time
    if n_samples <= 0:
        raise ValueError("Not enough frames for chosen lag_time.")

    # Work in sample index space: [0, n_samples)
    edges = np.linspace(0, n_samples, k + 1).astype(int)
    folds = []

    for i in range(k):
        v_start, v_end = edges[i], edges[i + 1]

        val_mask = np.zeros(n_samples, dtype=bool)
        val_mask[v_start:v_end] = True

        train_mask = ~val_mask

        if purge:
            # remove +/- lag_time samples around validation block from training
            purge_start = max(0, v_start - lag_time)
            purge_end = min(n_samples, v_end + lag_time)
            train_mask[purge_start:purge_end] = False

        folds.append((train_mask, val_mask))

    return folds


class DeepTICAWithEpochLogging(DeepTICA):
    """
    DeepTICA wrapper that adds epoch-level training loss logging.
    """
    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        # Log at epoch level for plotting training curves
        self.log("train_loss_epoch", loss, prog_bar=False, on_step=False, on_epoch=True)
        return loss


class DeepTICAWithAux(DeepTICA):
    """
    DeepTICA with an auxiliary loss that aligns CV1 with desorption:
      maximize corr(CV1, -dZ) and corr(CV1, -contacts)

    Assumes your input ordering:
      descriptor_cols = ['dZ', 'contacts', 'rg_lig', 'ree', 'asph_lig']
    so dz_idx=0, contacts_idx=1
    """
    def __init__(self, *args, aux_weight=1.0, dz_idx=0, contacts_idx=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_weight = float(aux_weight)
        self.dz_idx = int(dz_idx)
        self.contacts_idx = int(contacts_idx)

    @staticmethod
    def _pearson_corr(x, y, eps=1e-8):
        """
        Pearson correlation for two 1D tensors. Returns scalar in [-1,1].
        """
        x = x - x.mean()
        y = y - y.mean()
        vx = torch.sqrt(torch.mean(x * x) + eps)
        vy = torch.sqrt(torch.mean(y * y) + eps)
        return torch.mean(x * y) / (vx * vy + eps)

    def _aux_loss(self, batch):
        """
        Auxiliary loss: encourage CV1 to correlate with (-dZ) and (-contacts).
        Loss is (1 - corr)/2 averaged over the two targets (so minimizing increases corr).
        """
        x = batch["data"]  # shape (B, n_features)
        cv1 = self.forward_cv(x)[:, 0]  # (B,)

        dz = x[:, self.dz_idx]
        ct = x[:, self.contacts_idx]

        # We want CV1 to increase with "desorption-ness".
        # Since your dZ is negative and becomes more negative on desorption,
        # a common choice is to correlate with (-dZ) so "larger" means more desorbed.
        target_dz = -dz
        target_ct = -ct  # fewer contacts -> more desorbed => -contacts increases with desorption

        corr_dz = self._pearson_corr(cv1, target_dz)
        corr_ct = self._pearson_corr(cv1, target_ct)

        # Convert maximize-corr into minimize-loss
        loss_dz = 0.5 * (1.0 - corr_dz)
        loss_ct = 0.5 * (1.0 - corr_ct)
        aux = 0.5 * (loss_dz + loss_ct)

        return aux, corr_dz.detach(), corr_ct.detach()

    def training_step(self, batch, batch_idx):
        # base DeepTICA loss
        base_loss = super().training_step(batch, batch_idx)

        if getattr(self, "aux_weight", 0.0) > 0.0:
            aux, corr_dz, corr_ct = self._aux_loss(batch)
            total = base_loss + self.aux_weight * aux

            # Logging (both step and epoch level for training curves)
            self.log("train_loss_epoch", total, prog_bar=False, on_step=False, on_epoch=True)
            self.log("aux_loss", aux, prog_bar=True, on_step=True, on_epoch=True)
            self.log("corr_cv1_negdZ", corr_dz, prog_bar=False, on_step=True, on_epoch=True)
            self.log("corr_cv1_negcontacts", corr_ct, prog_bar=False, on_step=True, on_epoch=True)
            self.log("total_loss", total, prog_bar=True, on_step=True, on_epoch=True)

            return total
        else:
            # Log epoch-level training loss for standard DeepTICA
            self.log("train_loss_epoch", base_loss, prog_bar=False, on_step=False, on_epoch=True)

        return base_loss


def fit_one_split(X, config, descriptor_cols, train_mask_samples=None, val_mask_samples=None, verbose=True):
    """
    Train one model on a specified (train,val) mask split in sample space.
    If masks are None, falls back to DictModule random_split behavior.
    """
    lag_time = config["lag_time"]

    if train_mask_samples is None or val_mask_samples is None:
        # Fall back to original behavior (random split inside DictModule)
        dataset = create_timelagged_dataset_masked(X, lag_time=lag_time, mask_samples=None, logweights=None)

        datamodule = DictModule(
            dataset=dataset,
            lengths=[0.8, 0.2],
            batch_size=config["batch_size"],
            random_split=True,
            shuffle=True,
        )
    else:
        # Build explicit datasets for train and val to avoid leakage
        train_ds = create_timelagged_dataset_masked(X, lag_time=lag_time, mask_samples=train_mask_samples, logweights=None)
        val_ds   = create_timelagged_dataset_masked(X, lag_time=lag_time, mask_samples=val_mask_samples, logweights=None)


        datamodule = ExplicitDictDataModule(
        train_ds=train_ds,
        val_ds=val_ds,
        batch_size=config["batch_size"],
        num_workers=0,
        shuffle=True,
        )

    n_input = X.shape[1]
    layers = [n_input] + [config["hidden_size"]] * config["n_layers"] + [config["n_cvs"]]

    # Pick model class based on use_aux
    if args.use_aux:
        model = DeepTICAWithAux(
            layers=layers,
            n_cvs=config["n_cvs"],
            options={"nn": {"activation": "tanh"}},
            aux_weight=float(args.aux_weight),
            dz_idx=0,
            contacts_idx=1,
        )
    else:
        # Use wrapper that adds epoch-level logging
        model = DeepTICAWithEpochLogging(
            layers=layers,
            n_cvs=config["n_cvs"],
            options={"nn": {"activation": "tanh"}},
        )

    model.configure_optimizers = lambda: torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=1e-6,
    )

    run_name = f"lr{config['learning_rate']}_lag{config['lag_time']}_h{config['hidden_size']}_cvs{config['n_cvs']}"
    if args.use_aux:
        run_name += f"_aux{args.aux_weight}"

    logger = CSVLogger("training_logs", name=run_name)

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss_epoch",
        dirpath=f"checkpoints/{run_name}",
        filename="best-{epoch:02d}-{valid_loss_epoch:.6f}",
        save_top_k=1,
        mode="min",
    )

    early_stop = EarlyStopping(
        monitor="valid_loss_epoch",
        patience=40,
        min_delta=1e-4,
        mode="min",
    )

    trainer = L.Trainer(
        max_epochs=config["max_epochs"],
        accelerator="auto",
        devices=1,
        gradient_clip_val=1.0,
        enable_progress_bar=verbose,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop],
        enable_model_summary=False,
    )

    trainer.fit(model, datamodule)

    with torch.no_grad():
        evals = model.tica.evals.detach().cpu().numpy()
        X_tensor = torch.FloatTensor(X)
        cvs = model.forward_cv(X_tensor).detach().cpu().numpy()

    results = {
        "config": config,
        "model": model,
        "trainer": trainer,
        "logger": logger,
        "checkpoint": checkpoint_callback,
        "eigenvalues": evals,
        "cvs": cvs,
        "best_loss": float(checkpoint_callback.best_model_score) if checkpoint_callback.best_model_score else float("inf"),
        "epochs": int(trainer.current_epoch),
        "layers": layers,
        "descriptor_cols": descriptor_cols,
    }
    return results


def run_blocked_kfold_cv(X, config, k, lag_time, purge=True, verbose=False):
    """
    Runs blocked k-fold CV. Returns:
      - fold_summaries list
      - best_fold_idx (lowest val loss)
      - best_results (trained model results for that fold)
    """
    n_frames = len(X)
    folds = make_blocked_folds(n_frames=n_frames, k=k, lag_time=lag_time, purge=purge)
    fold_summaries = []

    best_idx = None
    best_loss = np.inf
    best_results = None

    for fi, (train_mask, val_mask) in enumerate(folds, 1):
        print(f"\n   [Fold {fi}/{k}] Training with blocked split "
              f"(purge={'ON' if purge else 'OFF'}, lag={lag_time})")

        res = fit_one_split(
            X=X,
            config=config,
            descriptor_cols=descriptor_cols,
            train_mask_samples=train_mask,
            val_mask_samples=val_mask,
            verbose=verbose
        )

        # Fold summary
        fold_info = {
            "fold": fi,
            "val_loss": float(res["best_loss"]),
            "epochs": int(res["epochs"]),
            "eigenvalues": res["eigenvalues"].tolist(),
            "cv1_corr_negdZ": None,
            "cv1_corr_negcontacts": None,
        }

        # Compute fold-level correlations to confirm desorption alignment
        cvs = res["cvs"]
        cv1 = cvs[:, 0]
        dz = X[:, 0]          # your order: dZ first
        ct = X[:, 1]          # contacts second

        fold_info["cv1_corr_negdZ"] = float(np.corrcoef(cv1, -dz)[0, 1])
        fold_info["cv1_corr_negcontacts"] = float(np.corrcoef(cv1, -ct)[0, 1])

        print(f"       Fold val_loss={fold_info['val_loss']:.6f} | "
              f"corr(CV1,-dZ)={fold_info['cv1_corr_negdZ']:+.3f} | "
              f"corr(CV1,-contacts)={fold_info['cv1_corr_negcontacts']:+.3f}")

        fold_summaries.append(fold_info)

        if fold_info["val_loss"] < best_loss:
            best_loss = fold_info["val_loss"]
            best_idx = fi
            best_results = res

    return fold_summaries, best_idx, best_results


# ==================== 5. PARAMETER SWEEP (OPTIONAL) ====================
# IMPORTANT: run sweep BEFORE final training so best_config is actually used.

if args.sweep:
    print("\n4. Running parameter sweep...")

    sweep_params = {
        "learning_rate": [1e-3, 5e-4, 1e-4, 5e-5],
        "lag_time": [25, 50, 100],
        "hidden_size": [32, 64, 128],
    }

    fixed_params = {
        "n_layers": args.layers,
        "n_cvs": args.n_cvs,
        "max_epochs": args.epochs,
        "batch_size": args.batch,
    }

    print(f"   Sweep parameters: {sweep_params}")
    print(f"   Fixed parameters: {fixed_params}")

    param_names = list(sweep_params.keys())
    combinations = list(product(*sweep_params.values()))
    print(f"   Total combinations: {len(combinations)}")

    sweep_results = []

    for i, combo in enumerate(combinations):
        config = {**fixed_params, **dict(zip(param_names, combo))}
        print(f"\n   [{i+1}/{len(combinations)}] Training: lr={config['learning_rate']}, "
              f"lag={config['lag_time']}, hidden={config['hidden_size']}")

        try:
            # Use fit_one_split, not train_model (train_model doesn't exist in this script)
            result = fit_one_split(
                X=X,
                config=config,
                descriptor_cols=descriptor_cols,
                train_mask_samples=None,
                val_mask_samples=None,
                verbose=False
            )

            valid = np.all(np.asarray(result["eigenvalues"]) > 0)

            sweep_results.append({
                "config": config,
                "best_loss": float(result["best_loss"]),
                "eigenvalues": result["eigenvalues"].tolist(),
                "cv_range": [float(result["cvs"][:, 0].min()), float(result["cvs"][:, 0].max())],
                "epochs": int(result["epochs"]),
                "valid": bool(valid),
            })

            print(f"       Loss: {result['best_loss']:.6f}, "
                  f"Eig1: {result['eigenvalues'][0]:.6f}, Valid: {'Yes' if valid else 'No'}")

        except Exception as e:
            print(f"       FAILED: {e}")
            sweep_results.append({
                "config": config,
                "best_loss": float("inf"),
                "eigenvalues": [],
                "cv_range": [0.0, 0.0],
                "epochs": 0,
                "valid": False,
                "error": str(e),
            })

    valid_results = [r for r in sweep_results if r["valid"] and np.isfinite(r["best_loss"])]

    if valid_results:
        best_result = min(valid_results, key=lambda x: x["best_loss"])
        best_config = best_result["config"]
        print("\n   Best configuration (lowest loss with valid eigenvalues):")
        print(f"   {best_config}")
        print(f"   Loss: {best_result['best_loss']:.6f}")
    else:
        print("\n   WARNING: No configurations produced valid eigenvalues!")
        print("   Using CLI defaults...")
        best_config = {
            "learning_rate": args.lr,
            "lag_time": args.lag,
            "hidden_size": args.hidden,
            "n_layers": args.layers,
            "n_cvs": args.n_cvs,
            "max_epochs": args.epochs,
            "batch_size": args.batch,
        }

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    with open(output_dir / "sweep_results.json", "w") as f:
        json.dump(sweep_results, f, indent=2, default=str)
    print(f"   Saved: {output_dir}/sweep_results.json")

else:
    best_config = {
        "learning_rate": args.lr,
        "lag_time": args.lag,
        "hidden_size": args.hidden,
        "n_layers": args.layers,
        "n_cvs": args.n_cvs,
        "max_epochs": args.epochs,
        "batch_size": args.batch,
    }


# ==================== 6. FINAL TRAINING / CV DRIVER ====================

print(f"\n5. Training mode:")
print(f"   Unweighted training: YES (logweights=None)")
print(f"   Aux loss enabled:    {'YES' if args.use_aux else 'NO'}")
if args.use_aux:
    print(f"   Aux weight:          {args.aux_weight}")
print(f"   Blocked k-fold:      {args.kfold}")
print(f"   Purge gap:           {'ON' if args.purge else 'OFF'}")

print("\n   Final config:")
print(f"   {best_config}")

if args.kfold and args.kfold >= 2:
    print("\n   Running BLOCKED k-fold cross validation...")
    fold_summaries, best_fold_idx, best_fold_results = run_blocked_kfold_cv(
        X=X,
        config=best_config,
        k=int(args.kfold),
        lag_time=int(best_config["lag_time"]),
        purge=bool(args.purge),
        verbose=False
    )

    # Save fold summaries
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    with open(output_dir / "kfold_summaries.json", "w") as f:
        json.dump(fold_summaries, f, indent=2)

    print(f"\n   Saved: {output_dir}/kfold_summaries.json")
    print(f"   Best fold: {best_fold_idx} (lowest val_loss)")

    # Use best fold's model as "results" downstream (visualizations/export)
    results = best_fold_results

else:
    print("\n   Running single training run (random 80/20 split inside DictModule)...")
    results = fit_one_split(
        X=X,
        config=best_config,
        descriptor_cols=descriptor_cols,
        train_mask_samples=None,
        val_mask_samples=None,
        verbose=True
    )

print("\n   Training complete!")
print(f"   Best validation loss: {results['best_loss']:.6f}")
print(f"   Epochs: {results['epochs']}")
print(f"   Eigenvalues: {results['eigenvalues']}")


# ==================== 7. HELPER FUNCTIONS FOR VISUALIZATION ====================

def compute_feature_importance(model, X_arr, descriptor_cols_used):
    """Gradient-based sensitivity analysis."""
    model.eval()
    X_tensor = torch.FloatTensor(X_arr)
    X_tensor.requires_grad = True

    cvs = model.forward_cv(X_tensor)
    importance = {}

    for cv_idx in range(cvs.shape[1]):
        model.zero_grad(set_to_none=True)
        cv_sum = cvs[:, cv_idx].sum()
        cv_sum.backward(retain_graph=True)

        grads = X_tensor.grad.abs().mean(dim=0).detach().cpu().numpy()
        X_tensor.grad.zero_()

        importance[f"CV{cv_idx+1}"] = {col: float(grads[i]) for i, col in enumerate(descriptor_cols_used)}

    return importance


def compute_feature_correlations(cvs, X_arr, descriptor_cols_used):
    """Pearson/Spearman correlations."""
    correlations = {"pearson": {}, "spearman": {}}
    for cv_idx in range(cvs.shape[1]):
        cv_name = f"CV{cv_idx+1}"
        correlations["pearson"][cv_name] = {}
        correlations["spearman"][cv_name] = {}
        for i, col in enumerate(descriptor_cols_used):
            r_p, p_p = stats.pearsonr(X_arr[:, i], cvs[:, cv_idx])
            r_s, p_s = stats.spearmanr(X_arr[:, i], cvs[:, cv_idx])
            correlations["pearson"][cv_name][col] = {"r": float(r_p), "p": float(p_p)}
            correlations["spearman"][cv_name][col] = {"r": float(r_s), "p": float(p_s)}
    return correlations


def rolling_mean(x, win):
    if win <= 1:
        return x
    win = min(win, len(x))
    kernel = np.ones(win, dtype=float) / float(win)
    return np.convolve(x, kernel, mode="valid")


def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def create_super_visualizations(results, X_arr, t_arr, colvar_df, descriptor_cols_used, output_dir: Path, dZ_idx, contacts_idx):
    """
    Creates ~15–20 plots. Some heavy plots are subsampled by args.plot_stride / args.pairplot_max.
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    cvs = results["cvs"]
    n_cvs = cvs.shape[1]
    n_feat = len(descriptor_cols_used)

    # Subsampling indices for heavy scatter plots
    stride = max(1, int(args.plot_stride))
    idx = np.arange(0, len(t_arr), stride, dtype=int)

    Xs = X_arr[idx]
    ts = t_arr[idx]
    cvss = cvs[idx]

    # For pairplot, cap points
    if len(idx) > args.pairplot_max:
        idx_pair = np.linspace(0, len(t_arr) - 1, args.pairplot_max).astype(int)
        Xp = X_arr[idx_pair]
        cvp = cvs[idx_pair]
    else:
        Xp = Xs
        cvp = cvss

    # 1) Training curves
    try:
        metrics_path = Path(results["logger"].log_dir) / "metrics.csv"
        metrics = pd.read_csv(metrics_path)

        plt.figure(figsize=(8, 4))

        # Training loss: try epoch-level first (new format), fall back to step-level
        train_plotted = False
        if "train_loss_epoch" in metrics.columns:
            # New format: epoch-level aggregated training loss
            train_data = metrics[metrics["train_loss_epoch"].notna()].copy()
            if len(train_data) > 0:
                plt.plot(train_data["epoch"], train_data["train_loss_epoch"],
                        label="train_loss", alpha=0.8, linewidth=1.5, marker='o', markersize=3)
                train_plotted = True

        if not train_plotted and "train_loss_step" in metrics.columns:
            # Old format: step-level, need to aggregate by epoch
            train_data = metrics[metrics["epoch"].notna()].copy()
            if len(train_data) > 0:
                train_grouped = train_data.groupby("epoch")["train_loss_step"].mean()
                plt.plot(train_grouped.index, train_grouped.values,
                        label="train_loss", alpha=0.8, linewidth=1.5, marker='o', markersize=3)

        # Validation loss: rows with empty epoch and valid_loss_step column
        if "valid_loss_step" in metrics.columns:
            valid_data = metrics[metrics["epoch"].isna()].copy()
            if len(valid_data) > 0 and "step" in valid_data.columns:
                # The step column for validation corresponds to epoch number
                plt.plot(valid_data["step"], valid_data["valid_loss_step"],
                        label="valid_loss", alpha=0.8, linewidth=1.5, marker='s', markersize=3)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training / Validation Loss")
        plt.legend()
        plt.grid(alpha=0.3)
        savefig(output_dir / "01_training_loss.png")
        print("   Saved: 01_training_loss.png")
    except Exception as e:
        print(f"   Warning: Could not plot training curves: {e}")

    # 2) Eigenvalues bar plot
    evals = results["eigenvalues"]
    plt.figure(figsize=(6, 4))
    plt.bar(np.arange(len(evals)), evals)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(np.arange(len(evals)), [f"CV{i+1}" for i in range(len(evals))])
    plt.ylabel("Eigenvalue")
    plt.title("DeepTICA Eigenvalues")
    plt.grid(alpha=0.3, axis="y")
    savefig(output_dir / "02_eigenvalues.png")
    print("   Saved: 02_eigenvalues.png")

    # 3) Feature distributions (hist+KDE) in one figure
    fig, axes = plt.subplots(n_feat, 1, figsize=(8, 2.2 * n_feat), sharex=False)
    if n_feat == 1:
        axes = [axes]
    for i, col in enumerate(descriptor_cols_used):
        ax = axes[i]
        sns.histplot(X_arr[:, i], bins=60, kde=True, ax=ax)
        ax.set_title(f"Distribution: {col}")
        ax.grid(alpha=0.2)
    savefig(output_dir / "03_feature_distributions.png")
    print("   Saved: 03_feature_distributions.png")

    # 4) Feature time series (raw + rolling mean)
    fig, axes = plt.subplots(n_feat, 1, figsize=(10, 2.2 * n_feat), sharex=True)
    if n_feat == 1:
        axes = [axes]
    for i, col in enumerate(descriptor_cols_used):
        ax = axes[i]
        ax.plot(t_arr / 1000.0, X_arr[:, i], linewidth=0.5, alpha=0.35)
        win = max(10, len(t_arr) // 200)
        rm = rolling_mean(X_arr[:, i], win)
        t_rm = t_arr[win // 2: win // 2 + len(rm)]
        ax.plot(t_rm / 1000.0, rm, linewidth=1.8)
        ax.set_ylabel(col)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Time (ns)")
    fig.suptitle("Feature Time Series (raw + rolling mean)", y=1.01)
    savefig(output_dir / "04_feature_timeseries.png")
    print("   Saved: 04_feature_timeseries.png")

    # 5) Feature-feature correlation heatmap
    plt.figure(figsize=(7, 6))
    corr = np.corrcoef(X_arr.T)
    sns.heatmap(corr, annot=True, fmt=".2f", xticklabels=descriptor_cols_used, yticklabels=descriptor_cols_used,
                cmap="RdBu_r", center=0, vmin=-1, vmax=1)
    plt.title("Feature-Feature Correlation (Pearson)")
    savefig(output_dir / "05_feature_corr_heatmap.png")
    print("   Saved: 05_feature_corr_heatmap.png")

    # 6) Pairplot (features + CV1) — capped points
    try:
        df_pair = pd.DataFrame(Xp, columns=descriptor_cols_used)
        df_pair["CV1"] = cvp[:, 0]
        sns.pairplot(df_pair, corner=True, plot_kws={"s": 8, "alpha": 0.2})
        plt.suptitle("Pairplot: Features + CV1 (subsampled)", y=1.02)
        plt.savefig(output_dir / "06_pairplot_features_cv1.png", dpi=160, bbox_inches="tight")
        plt.close()
        print("   Saved: 06_pairplot_features_cv1.png")
    except Exception as e:
        print(f"   Warning: Pairplot failed (likely memory): {e}")

    # 7) CV time series
    fig, axes = plt.subplots(n_cvs, 1, figsize=(10, 2.3 * n_cvs), sharex=True)
    if n_cvs == 1:
        axes = [axes]
    for k in range(n_cvs):
        ax = axes[k]
        ax.scatter(ts / 1000.0, cvss[:, k], s=2, alpha=0.35)
        ax.set_ylabel(f"CV{k+1}")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Time (ns)")
    fig.suptitle("DeepTICA CV Time Series (subsampled)", y=1.01)
    savefig(output_dir / "07_cv_timeseries.png")
    print("   Saved: 07_cv_timeseries.png")

    # 8) CV distributions
    fig, axes = plt.subplots(n_cvs, 1, figsize=(8, 2.2 * n_cvs))
    if n_cvs == 1:
        axes = [axes]
    for k in range(n_cvs):
        ax = axes[k]
        sns.histplot(cvs[:, k], bins=80, kde=True, ax=ax)
        ax.set_title(f"Distribution: CV{k+1}")
        ax.grid(alpha=0.2)
    savefig(output_dir / "08_cv_distributions.png")
    print("   Saved: 08_cv_distributions.png")

    # 9) CV vs each feature (scatter, subsampled)
    for k in range(min(n_cvs, 2)):  # keep it readable; you can extend if desired
        fig, axes = plt.subplots(1, n_feat, figsize=(4.2 * n_feat, 3.8), sharey=True)
        if n_feat == 1:
            axes = [axes]
        for i, col in enumerate(descriptor_cols_used):
            ax = axes[i]
            ax.scatter(Xs[:, i], cvss[:, k], s=2, alpha=0.25)
            ax.set_xlabel(col)
            if i == 0:
                ax.set_ylabel(f"CV{k+1}")
            ax.grid(alpha=0.2)
        fig.suptitle(f"CV{k+1} vs Features (subsampled)", y=1.03)
        savefig(output_dir / f"09_cv{k+1}_vs_features.png")
        print(f"   Saved: 09_cv{k+1}_vs_features.png")

    # 10) Correlation heatmaps: features vs CVs (Pearson/Spearman)
    correlations = compute_feature_correlations(cvs, X_arr, descriptor_cols_used)
    pearson_mat = np.array([[correlations["pearson"][f"CV{cv+1}"][col]["r"] for col in descriptor_cols_used]
                            for cv in range(n_cvs)])
    spearman_mat = np.array([[correlations["spearman"][f"CV{cv+1}"][col]["r"] for col in descriptor_cols_used]
                             for cv in range(n_cvs)])

    plt.figure(figsize=(1.5 + 1.2 * n_feat, 1.0 + 0.7 * n_cvs))
    sns.heatmap(pearson_mat, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                xticklabels=descriptor_cols_used, yticklabels=[f"CV{i+1}" for i in range(n_cvs)],
                vmin=-1, vmax=1)
    plt.title("Pearson r: CVs vs Features")
    plt.xticks(rotation=45, ha="right")
    savefig(output_dir / "10_corr_cv_feature_pearson.png")
    print("   Saved: 10_corr_cv_feature_pearson.png")

    plt.figure(figsize=(1.5 + 1.2 * n_feat, 1.0 + 0.7 * n_cvs))
    sns.heatmap(spearman_mat, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                xticklabels=descriptor_cols_used, yticklabels=[f"CV{i+1}" for i in range(n_cvs)],
                vmin=-1, vmax=1)
    plt.title("Spearman ρ: CVs vs Features")
    plt.xticks(rotation=45, ha="right")
    savefig(output_dir / "11_corr_cv_feature_spearman.png")
    print("   Saved: 11_corr_cv_feature_spearman.png")

    # 11) Gradient-based feature importance bars
    importance = compute_feature_importance(results["model"], X_arr, descriptor_cols_used)

    for k in range(n_cvs):
        imp = importance[f"CV{k+1}"]
        names = list(imp.keys())
        vals = np.array([imp[n] for n in names], dtype=float)
        vals = vals / (vals.sum() + 1e-12) * 100.0

        order = np.argsort(vals)[::-1]
        plt.figure(figsize=(7, 4))
        plt.bar([names[i] for i in order], vals[order])
        plt.ylabel("Relative importance (%)")
        plt.title(f"Feature Importance (Gradient): CV{k+1}")
        plt.xticks(rotation=45, ha="right")
        plt.grid(alpha=0.2, axis="y")
        savefig(output_dir / f"12_feature_importance_cv{k+1}.png")
        print(f"   Saved: 12_feature_importance_cv{k+1}.png")

    # 12) 2D density / FES-style plots (hist2d -> -kT ln P) for desorption axes
    kb = 0.008314  # kJ/(mol K)
    temp = 300.0

    def fes2d(x, y, bins=70):
        H, xedges, yedges = np.histogram2d(x, y, bins=bins, density=True)
        H = np.ma.masked_where(H <= 0, H)
        F = -kb * temp * np.log(H)
        F = F - np.nanmin(F)
        return F.T, xedges, yedges

    # CV1 vs dZ
    F, xe, ye = fes2d(X_arr[:, dZ_idx], cvs[:, 0], bins=80)
    plt.figure(figsize=(6, 5))
    plt.imshow(F, origin="lower", aspect="auto",
               extent=[xe[0], xe[-1], ye[0], ye[-1]])
    plt.xlabel("dZ")
    plt.ylabel("CV1")
    plt.title("FES: CV1 vs dZ")
    plt.colorbar(label="F (kJ/mol)")
    savefig(output_dir / "13_fes_cv1_vs_dZ.png")
    print("   Saved: 13_fes_cv1_vs_dZ.png")

    # CV1 vs contacts
    F, xe, ye = fes2d(X_arr[:, contacts_idx], cvs[:, 0], bins=80)
    plt.figure(figsize=(6, 5))
    plt.imshow(F, origin="lower", aspect="auto",
               extent=[xe[0], xe[-1], ye[0], ye[-1]])
    plt.xlabel("contacts")
    plt.ylabel("CV1")
    plt.title("FES: CV1 vs contacts")
    plt.colorbar(label="F (kJ/mol)")
    savefig(output_dir / "14_fes_cv1_vs_contacts.png")
    print("   Saved: 14_fes_cv1_vs_contacts.png")

    # dZ vs contacts
    F, xe, ye = fes2d(X_arr[:, dZ_idx], X_arr[:, contacts_idx], bins=80)
    plt.figure(figsize=(6, 5))
    plt.imshow(F, origin="lower", aspect="auto",
               extent=[xe[0], xe[-1], ye[0], ye[-1]])
    plt.xlabel("dZ")
    plt.ylabel("contacts")
    plt.title("FES: dZ vs contacts")
    plt.colorbar(label="F (kJ/mol)")
    savefig(output_dir / "15_fes_dZ_vs_contacts.png")
    print("   Saved: 15_fes_dZ_vs_contacts.png")

    # 13) Hexbin plots (dense scatter alternative)
    gs = int(args.hexbin_gridsize)
    plt.figure(figsize=(6, 5))
    plt.hexbin(Xs[:, dZ_idx], Xs[:, contacts_idx], gridsize=gs, mincnt=1)
    plt.xlabel("dZ")
    plt.ylabel("contacts")
    plt.title("Hexbin: dZ vs contacts (subsampled)")
    plt.colorbar(label="count")
    savefig(output_dir / "16_hexbin_dZ_vs_contacts.png")
    print("   Saved: 16_hexbin_dZ_vs_contacts.png")

    plt.figure(figsize=(6, 5))
    plt.hexbin(cvs[idx, 0], Xs[:, contacts_idx], gridsize=gs, mincnt=1)
    plt.xlabel("CV1")
    plt.ylabel("contacts")
    plt.title("Hexbin: CV1 vs contacts (subsampled)")
    plt.colorbar(label="count")
    savefig(output_dir / "17_hexbin_cv1_vs_contacts.png")
    print("   Saved: 17_hexbin_cv1_vs_contacts.png")

    # 14) CV phase portrait (CV1 vs CV2) if available
    if n_cvs >= 2:
        plt.figure(figsize=(6, 5))
        plt.scatter(cvss[:, 0], cvss[:, 1], s=2, alpha=0.25)
        plt.xlabel("CV1")
        plt.ylabel("CV2")
        plt.title("CV Phase Space (subsampled)")
        plt.grid(alpha=0.2)
        savefig(output_dir / "18_cv1_vs_cv2_scatter.png")
        print("   Saved: 18_cv1_vs_cv2_scatter.png")

        F, xe, ye = fes2d(cvs[:, 0], cvs[:, 1], bins=90)
        plt.figure(figsize=(6, 5))
        plt.imshow(F, origin="lower", aspect="auto",
                   extent=[xe[0], xe[-1], ye[0], ye[-1]])
        plt.xlabel("CV1")
        plt.ylabel("CV2")
        plt.title("FES: CV1 vs CV2")
        plt.colorbar(label="F (kJ/mol)")
        savefig(output_dir / "19_fes_cv1_vs_cv2.png")
        print("   Saved: 19_fes_cv1_vs_cv2.png")

    # 15) Rolling correlation of CV1 with dZ and contacts (over time)
    # (This is a great diagnostic for “does CV1 track desorption consistently?”)
    win = max(2000, len(t_arr) // 50)
    if len(t_arr) > win + 50:
        def rolling_corr(a, b, w):
            out = np.empty(len(a) - w + 1, dtype=float)
            for i in range(len(out)):
                out[i] = np.corrcoef(a[i:i+w], b[i:i+w])[0, 1]
            return out

        rc_dz = rolling_corr(cvs[:, 0], X_arr[:, dZ_idx], win)
        rc_ct = rolling_corr(cvs[:, 0], X_arr[:, contacts_idx], win)
        t_mid = t_arr[win//2: win//2 + len(rc_dz)] / 1000.0

        plt.figure(figsize=(10, 4))
        plt.plot(t_mid, rc_dz, label="corr(CV1, dZ)")
        plt.plot(t_mid, rc_ct, label="corr(CV1, contacts)")
        plt.axhline(0, linestyle="--", linewidth=1)
        plt.xlabel("Time (ns)")
        plt.ylabel("Rolling correlation")
        plt.title(f"Rolling Correlation (window={win} frames)")
        plt.legend()
        plt.grid(alpha=0.2)
        savefig(output_dir / "20_rolling_corr_cv1_desorption.png")
        print("   Saved: 20_rolling_corr_cv1_desorption.png")

    return importance, correlations


# ==================== 8. EXTENSIVE VISUALIZATION (15–20 plots) ====================
print("\n6. Creating super comprehensive visualizations...")

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True)

importance, correlations = create_super_visualizations(
    results=results,
    X_arr=X,
    t_arr=t,
    colvar_df=colvar,
    descriptor_cols_used=descriptor_cols,
    output_dir=output_dir,
    dZ_idx=dZ_idx,
    contacts_idx=contacts_idx,
)


# ==================== 8. EXPORT MODEL FOR PLUMED ====================
print("\n7. Exporting model for PLUMED...")

model = results["model"]
cvs = results["cvs"]
n_input = X.shape[1]

class ModelForExport(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.norm_in = original_model.norm_in
        self.nn = original_model.nn

    def forward(self, x):
        x = self.norm_in(x)
        x = self.nn(x)
        return x

export_model = ModelForExport(model)
export_model.eval()

dummy_input = torch.randn(1, n_input)
with torch.no_grad():
    traced_model = torch.jit.trace(export_model, dummy_input)
    test_output = traced_model(dummy_input)
    print("   Model traced successfully")
    print(f"   Test output shape: {test_output.shape}")

model_path = output_dir / "model.ptc"
traced_model.save(str(model_path))
print(f"   Model saved: {model_path}")


# ==================== 9. SAVE METADATA ====================
print("\n8. Saving comprehensive metadata...")

metadata = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "training_data": {
        "source": args.colvar,
        "frames": int(len(colvar)),
        "duration_ns": float((colvar["time"].max() - colvar["time"].min()) / 1000.0),
        "unweighted_training": True,
    },
    "descriptors": {
        "descriptor_order": descriptor_cols,
        "note": "No feature scaling applied. Training is unweighted.",
    },
    "model": {
        "architecture": results["layers"],
        "n_cvs": int(best_config["n_cvs"]),
        "learning_rate": float(best_config["learning_rate"]),
        "lag_time": int(best_config["lag_time"]),
        "batch_size": int(best_config["batch_size"]),
        "epochs_trained": int(results["epochs"]),
        "best_loss": float(results["best_loss"]),
    },
    "eigenvalues": results["eigenvalues"].tolist(),
    "feature_stats": feature_stats,
    "feature_importance": importance,
    "correlations": correlations,
    "cv_range": {
        "min": float(cvs[:, 0].min()),
        "max": float(cvs[:, 0].max()),
        "mean": float(cvs[:, 0].mean()),
        "std": float(cvs[:, 0].std()),
    },
}

with open(output_dir / "model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2, default=str)

print("   Saved: model_metadata.json")


# ==================== 10. CREATE PRODUCTION PLUMED ====================
print("\n9. Creating production PLUMED file...")

cv_min = float(cvs[:, 0].min() - 0.5)
cv_max = float(cvs[:, 0].max() + 0.5)

# IMPORTANT: The descriptors here must be defined as PLUMED variables with these names.
# And the order in PYTORCH_MODEL ARG must match descriptor_cols in Python.
plumed_production = f"""# PLUMED input for Production Run with DeepTICA CV
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Training loss: {results['best_loss']:.6f}
# CV1 eigenvalue: {results['eigenvalues'][0]:.6f}

UNITS LENGTH=nm ENERGY=kj/mol TIME=ps

# ==================== DEFINE GROUPS ====================
PE: GROUP NDX_FILE=PE-PEUS-HOH.ndx NDX_GROUP=PE
LIG: GROUP NDX_FILE=PE-PEUS-HOH.ndx NDX_GROUP=PIS_PUS_PTS
HOH: GROUP NDX_FILE=PE-PEUS-HOH.ndx NDX_GROUP=W
WHOLEMOLECULES ENTITY0=PE,LIG

COM_PE: COM ATOMS=PE
COM_LIG: COM ATOMS=LIG

# ==================== COMPUTE DESCRIPTORS ====================
rg_lig: GYRATION TYPE=RADIUS ATOMS=LIG
asph_lig: GYRATION TYPE=ASPHERICITY ATOMS=LIG

dist_components: DISTANCE ATOMS=COM_LIG,COM_PE COMPONENTS
dZ: COMBINE ARG=dist_components.z PERIODIC=NO

FIRST_BEAD: GROUP ATOMS=17001
LAST_BEAD: GROUP ATOMS=17495
ree: DISTANCE ATOMS=FIRST_BEAD,LAST_BEAD

contacts: COORDINATION GROUPA=LIG GROUPB=PE R_0=0.6 NN=6 MM=12

# ==================== DEEPTICA CV ====================
# No feature scaling applied - raw descriptors are used directly
deep: PYTORCH_MODEL FILE=model.ptc ARG=dZ,contacts,rg_lig,ree,asph_lig

# If n_cvs=1 you only have deep.node-0; if n_cvs=2 you have deep.node-0 and deep.node-1.
cv1: COMBINE ARG=deep.node-0 PERIODIC=NO
"""

# If you trained 2 CVs, define cv2 and optionally bias it or print it
if best_config["n_cvs"] >= 2:
    plumed_production += """cv2: COMBINE ARG=deep.node-1 PERIODIC=NO
"""

plumed_production += f"""
# ==================== OPES ENHANCED SAMPLING ====================
opes: OPES_METAD ...
    ARG=cv1
    PACE=200
    SIGMA=0.5
    SIGMA_MIN=0.2
    FILE=HILLS_production
    BARRIER=200
    BIASFACTOR=1000
...

# ==================== OUTPUT ====================
PRINT STRIDE=100 FILE=COLVAR_PRODUCTION ARG=dZ,contacts,rg_lig,ree,asph_lig,cv1{",cv2" if best_config["n_cvs"]>=2 else ""},opes.bias FMT=%12.6f

ENDPLUMED
"""

plumed_path = output_dir / "plumed-production.dat"
with open(plumed_path, "w") as f:
    f.write(plumed_production)

print(f"   Saved: {plumed_path}")


# ==================== 11. SUMMARY ====================
print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)

print("\nDescriptor order used in training (and in PYTORCH_MODEL ARG order):")
for i, c in enumerate(descriptor_cols, 1):
    print(f"  {i}. {c}")

print("\nNo feature scaling applied.")
print("Training mode: UNWEIGHTED (logweights=None)")

print("\nFeature Importance (CV1):")
imp_cv1 = importance["CV1"]
sorted_imp = sorted(imp_cv1.items(), key=lambda x: x[1], reverse=True)
total_imp = sum(imp_cv1.values()) + 1e-12
for name, val in sorted_imp:
    pct = 100.0 * val / total_imp
    bar = "#" * int(pct / 2)
    print(f"  {name:15s}: {pct:5.1f}% {bar}")

print("\nTop correlations with CV1 (Pearson):")
for name, data in sorted(correlations["pearson"]["CV1"].items(),
                         key=lambda x: abs(x[1]["r"]), reverse=True)[:5]:
    print(f"  {name:15s}: r = {data['r']:+.3f}  (p={data['p']:.2e})")

print(f"""
Output Files ({output_dir}/):
  - model.ptc                         : Traced model for PLUMED
  - model_metadata.json               : Training + analysis metadata
  - plumed-production.dat             : Production PLUMED input
  - 01_training_loss.png              : Train/valid loss curves
  - 02_eigenvalues.png                : TICA eigenvalues
  - 03_feature_distributions.png      : Feature hist/KDE
  - 04_feature_timeseries.png         : Feature time series + rolling mean
  - 05_feature_corr_heatmap.png       : Feature-feature correlation
  - 06_pairplot_features_cv1.png      : Pairplot (subsampled/capped)
  - 07_cv_timeseries.png              : CV time series
  - 08_cv_distributions.png           : CV hist/KDE
  - 09_cv*_vs_features.png            : CV vs feature scatters
  - 10_corr_cv_feature_pearson.png    : Pearson CV-feature heatmap
  - 11_corr_cv_feature_spearman.png   : Spearman CV-feature heatmap
  - 12_feature_importance_cv*.png     : Gradient feature importances
  - 13_fes_cv1_vs_dZ.png              : FES-like CV1 vs dZ
  - 14_fes_cv1_vs_contacts.png        : FES-like CV1 vs contacts
  - 15_fes_dZ_vs_contacts.png         : FES-like dZ vs contacts
  - 16_hexbin_dZ_vs_contacts.png      : Hexbin density
  - 17_hexbin_cv1_vs_contacts.png     : Hexbin density
  - 18_cv1_vs_cv2_scatter.png         : CV phase scatter (if n_cvs>=2)
  - 19_fes_cv1_vs_cv2.png             : FES-like CV1 vs CV2 (if n_cvs>=2)
  - 20_rolling_corr_cv1_desorption.png: Rolling corr(CV1,dZ/contacts)
""")

if args.sweep:
    print("  - sweep_results.json            : Parameter sweep results")

print("=" * 80)
