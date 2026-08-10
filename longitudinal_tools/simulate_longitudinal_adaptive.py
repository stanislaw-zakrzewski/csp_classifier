"""
Longitudinal Adaptive & Static Simulation Engine
=================================================

Performs:
1. Static Zero-Shot Simulation: Evaluates models on test sessions without updating weights.
2. Adaptive Online Simulation: Trial-by-trial streaming evaluation with Head-Only adaptation
   micro-batches (step I=8, buffer M=64) on test sessions.

Logs trial-level correctness, binned time trajectories (Bin 0 to Bin 9), and summary metrics.
"""

import os
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def evaluate_static_model(
    model_obj: object,
    model_name: str,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> dict[str, float]:
    """Evaluate a static (frozen) model on X_test, y_test."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_name in ["csp_lda", "cov_tgsp"]:
        if model_name == "csp_lda":
            X_feats = model_obj["csp"].transform(X_test)
            preds = model_obj["classifier"].predict(X_feats)
        else:
            X_cov = model_obj["cov"].transform(X_test)
            X_ts = model_obj["ts"].transform(X_cov)
            preds = model_obj["classifier"].predict(X_ts)
        acc = float(np.mean(preds == y_test))
        return {"accuracy": acc, "n_trials": len(y_test), "correct": int(np.sum(preds == y_test))}

    elif isinstance(model_obj, nn.Module):
        model_obj.eval()
        model_obj.to(device)
        X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model_obj(X_t)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        acc = float(np.mean(preds == y_test))
        return {"accuracy": acc, "n_trials": len(y_test), "correct": int(np.sum(preds == y_test))}

    else:
        raise TypeError(f"Unsupported model type for '{model_name}': {type(model_obj)}")


def simulate_adaptive_online(
    model_obj: object,
    model_name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    step_size: int = 8,
    buffer_size: int = 64,
    lr: float = 1e-3,
    adapt_epochs: int = 2
) -> pd.DataFrame:
    """
    Simulate online streaming adaptation trial-by-trial on a test session.
    Returns DataFrame containing columns: ['Trial', 'Bin', 'True_Label', 'Pred_Label', 'Is_Correct'].
    """
    n_trials = len(y_test)
    records = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if isinstance(model_obj, nn.Module):
        # Create an adaptive copy of the PyTorch model
        adaptive_model = copy.deepcopy(model_obj)
        adaptive_model.to(device)

        # Freeze backbone for Head-Only adaptation
        for name, param in adaptive_model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

        head_params = [p for p in adaptive_model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(head_params, lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        history_X, history_y = [], []

        for i in range(n_trials):
            trial_x = X_test[i:i+1]
            trial_y = y_test[i:i+1]

            # 1. Predict zero-shot on incoming trial
            adaptive_model.eval()
            with torch.no_grad():
                tx_t = torch.tensor(trial_x, dtype=torch.float32).to(device)
                logits = adaptive_model(tx_t)
                pred = int(torch.argmax(logits, dim=1).cpu().numpy()[0])

            is_correct = int(pred == trial_y[0])

            # Bin index 0..9 across session
            bin_idx = min(9, int(10 * i / max(1, n_trials)))

            records.append({
                "Trial": i,
                "Bin": bin_idx,
                "True_Label": int(trial_y[0]),
                "Pred_Label": pred,
                "Is_Correct": is_correct
            })

            # Store in history buffer
            history_X.append(trial_x[0])
            history_y.append(trial_y[0])

            # 2. Perform micro-batch head adaptation every step_size trials
            if (i + 1) % step_size == 0 and len(history_X) >= step_size:
                buf_X = np.array(history_X[-buffer_size:])
                buf_y = np.array(history_y[-buffer_size:])

                adaptive_model.train()
                b_X_t = torch.tensor(buf_X, dtype=torch.float32).to(device)
                b_y_t = torch.tensor(buf_y, dtype=torch.long).to(device)

                for _ in range(adapt_epochs):
                    optimizer.zero_grad()
                    out = adaptive_model(b_X_t)
                    loss = criterion(out, b_y_t)
                    loss.backward()
                    optimizer.step()

    else:
        # Classical model (static simulation per trial if no online adaptation wrapper)
        if model_name == "csp_lda":
            X_feats = model_obj["csp"].transform(X_test)
            preds = model_obj["classifier"].predict(X_feats)
        else:
            X_cov = model_obj["cov"].transform(X_test)
            X_ts = model_obj["ts"].transform(X_cov)
            preds = model_obj["classifier"].predict(X_ts)

        for i in range(n_trials):
            pred = int(preds[i])
            is_correct = int(pred == y_test[i])
            bin_idx = min(9, int(10 * i / max(1, n_trials)))
            records.append({
                "Trial": i,
                "Bin": bin_idx,
                "True_Label": int(y_test[i]),
                "Pred_Label": pred,
                "Is_Correct": is_correct
            })

    return pd.DataFrame(records)


def run_full_session_evaluations(
    models_dict: dict[str, object],
    test_sessions_dict: dict[str, tuple[np.ndarray, np.ndarray]],
    k_sessions: int,
    train_modality: str,
    test_modality: str,
    output_dir: str = "simulation_results/longitudinal"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run static and adaptive evaluation for all 6 models across all test sessions.
    Returns (summary_df, detailed_df).
    """
    os.makedirs(output_dir, exist_ok=True)
    summary_rows = []
    detailed_dfs = []

    for model_name, model_obj in models_dict.items():
        for session_name, (X_test, y_test) in test_sessions_dict.items():

            # 1. Static Evaluation
            static_res = evaluate_static_model(model_obj, model_name, X_test, y_test)

            # 2. Adaptive Evaluation
            adaptive_df = simulate_adaptive_online(model_obj, model_name, X_test, y_test)
            adaptive_acc = float(adaptive_df["Is_Correct"].mean())

            # Log summary row
            summary_rows.append({
                "k_train_sessions": k_sessions,
                "train_modality": train_modality,
                "test_modality": test_modality,
                "model": model_name,
                "test_session": session_name,
                "static_accuracy": static_res["accuracy"],
                "adaptive_accuracy": adaptive_acc,
                "delta_adaptive_minus_static": adaptive_acc - static_res["accuracy"],
                "n_trials": len(y_test)
            })

            # Tag detailed dataframe
            adaptive_df["k_train_sessions"] = k_sessions
            adaptive_df["train_modality"] = train_modality
            adaptive_df["test_modality"] = test_modality
            adaptive_df["model"] = model_name
            adaptive_df["test_session"] = session_name
            detailed_dfs.append(adaptive_df)

    summary_df = pd.DataFrame(summary_rows)
    detailed_df = pd.concat(detailed_dfs, ignore_index=True)

    # Save CSVs
    csv_sum_path = os.path.join(output_dir, f"summary_k{k_sessions}_{train_modality}_to_{test_modality}.csv")
    csv_det_path = os.path.join(output_dir, f"detailed_k{k_sessions}_{train_modality}_to_{test_modality}.csv")

    summary_df.to_csv(csv_sum_path, index=False)
    detailed_df.to_csv(csv_det_path, index=False)

    return summary_df, detailed_df
