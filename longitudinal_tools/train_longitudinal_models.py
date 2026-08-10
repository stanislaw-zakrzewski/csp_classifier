"""
Longitudinal Model Trainer Engine for Experiment 10
===================================================

Trains/Fine-tunes the 6 required classification approaches on k training sessions:
1. CSP-LDA (Classic baseline)
2. tGSP-Cov / COV-TGSP (Classic Riemannian baseline)
3. EEGNet Scratch (Trained only on longitudinal training sessions)
4. ATCNet Scratch (Trained only on longitudinal training sessions)
5. EEGNet Cluster Pretrained + Fine-Tuned (Fine-tuned from MOABB GNN cluster checkpoint)
6. ATCNet Cluster Pretrained + Fine-Tuned (Fine-tuned from MOABB GNN cluster checkpoint)
"""

import os
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from eegnet_tools.eegnet_model import EEGNet
from atcnet_tools.atcnet_model import ATCNet


def train_classical_csp_lda(X_train: np.ndarray, y_train: np.ndarray) -> object:
    """Fit CSP (6 components) + LDA pipeline."""
    csp = CSP(n_components=6, log=True, norm_trace=False)
    X_csp = csp.fit_transform(X_train, y_train)
    lda = LDA()
    lda.fit(X_csp, y_train)
    return {"csp": csp, "classifier": lda, "type": "CSP_LDA"}


def train_classical_cov_tgsp(X_train: np.ndarray, y_train: np.ndarray) -> object:
    """Fit Covariances + Tangent Space + Logistic Regression pipeline."""
    cov = Covariances(estimator='oas')
    X_cov = cov.fit_transform(X_train)
    ts = TangentSpace(metric='riemann')
    X_ts = ts.fit_transform(X_cov)
    clf = LogisticRegression(C=1.0, max_iter=500)
    clf.fit(X_ts, y_train)
    return {"cov": cov, "ts": ts, "classifier": clf, "type": "COV_TGSP"}


def train_pytorch_scratch_model(
    model_type: str,  # "EEGNet" or "ATCNet"
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> nn.Module:
    """Train EEGNet or ATCNet from scratch on longitudinal dataset."""
    chans = X_train.shape[1]
    samples = X_train.shape[2]

    if model_type == "EEGNet":
        model = EEGNet(n_classes=2, channels=chans, samples=samples, F1=8, D=2, F2=16, kernel_length=64, dropout_rate=0.25)
    elif model_type == "ATCNet":
        model = ATCNet(n_classes=2, channels=chans, samples=samples, dropout_rate=0.25)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Tensor dataset
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

    return model


def fine_tune_cluster_model(
    model_type: str,  # "EEGNet" or "ATCNet"
    pretrained_path: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 25,
    lr: float = 1e-4,
    batch_size: int = 32,
    head_only: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> nn.Module:
    """
    Load pre-trained cluster model weights and fine-tune on longitudinal training sessions.
    If head_only=True, freeze feature extractor backbone and only update classifier head.
    """
    chans = X_train.shape[1]
    samples = X_train.shape[2]

    if model_type == "EEGNet":
        model = EEGNet(n_classes=2, channels=chans, samples=samples, F1=8, D=2, F2=16, kernel_length=64, dropout_rate=0.25)
    elif model_type == "ATCNet":
        model = ATCNet(n_classes=2, channels=chans, samples=samples, dropout_rate=0.25)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")

    if os.path.exists(pretrained_path):
        state_dict = torch.load(pretrained_path, map_location=device, weights_only=True)
        # Adapt classifier weight size if mismatch occurs
        if "classifier.weight" in state_dict and state_dict["classifier.weight"].shape != model.classifier.weight.shape:
            # Reinitialize classifier in state_dict to match target model shape
            del state_dict["classifier.weight"]
            del state_dict["classifier.bias"]
        model.load_state_dict(state_dict, strict=False)

    model.to(device)

    if head_only:
        # Freeze backbone parameters, update only linear classifier
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params_to_update, lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    return model


def find_default_pretrained_cluster_checkpoint(model_type: str, dataset_ref: str = "Dreyer2023") -> str:
    """Find a pre-trained cluster checkpoint file for model_type."""
    base_dir = f"trained_pipelines/{model_type.lower()}/{dataset_ref}/cluster_mode_0"
    target_file = f"{model_type}_cluster_model.pt"
    fpath = os.path.join(base_dir, target_file)
    if os.path.exists(fpath):
        return fpath
    # Search recursively if default not found
    search_pattern = f"trained_pipelines/{model_type.lower()}/**/{model_type}_cluster_model.pt"
    matches = glob.glob(search_pattern, recursive=True)
    if matches:
        return matches[0]
    return ""


def train_all_longitudinal_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k_sessions: int,
    output_dir: str = "trained_pipelines/longitudinal"
) -> dict[str, object]:
    """
    Train and return dictionary containing all 6 trained model instances:
    1. 'csp_lda'
    2. 'cov_tgsp'
    3. 'eegnet_scratch'
    4. 'atcnet_scratch'
    5. 'eegnet_cluster_finetuned'
    6. 'atcnet_cluster_finetuned'
    """
    os.makedirs(output_dir, exist_ok=True)
    save_k_dir = os.path.join(output_dir, f"k_sessions_{k_sessions}")
    os.makedirs(save_k_dir, exist_ok=True)

    models_dict = {}

    # 1. Classical CSP-LDA
    models_dict['csp_lda'] = train_classical_csp_lda(X_train, y_train)

    # 2. Classical COV-TGSP
    models_dict['cov_tgsp'] = train_classical_cov_tgsp(X_train, y_train)

    # 3. EEGNet Scratch
    models_dict['eegnet_scratch'] = train_pytorch_scratch_model("EEGNet", X_train, y_train, epochs=50)

    # 4. ATCNet Scratch
    models_dict['atcnet_scratch'] = train_pytorch_scratch_model("ATCNet", X_train, y_train, epochs=50)

    # 5. EEGNet Cluster Pretrained + Fine-Tuned
    eegnet_ckpt = find_default_pretrained_cluster_checkpoint("EEGNet")
    models_dict['eegnet_cluster_finetuned'] = fine_tune_cluster_model(
        "EEGNet", eegnet_ckpt, X_train, y_train, epochs=40, lr=5e-4, head_only=False
    )

    # 6. ATCNet Cluster Pretrained + Fine-Tuned
    atcnet_ckpt = find_default_pretrained_cluster_checkpoint("ATCNet")
    models_dict['atcnet_cluster_finetuned'] = fine_tune_cluster_model(
        "ATCNet", atcnet_ckpt, X_train, y_train, epochs=40, lr=5e-4, head_only=False
    )

    return models_dict

