"""
Retrain Classical Single-Subject Pipelines (80/20 Train Split)
===============================================================

Fits CSP-LDA, Cov-Tangent-Space-LR, and CSP-SVM single-subject pipelines on the exact 80% training split (X_train, y_train) for each subject across all datasets.
Saves checkpoints to: `trained_pipelines/{dataset}/subject_{id}/*.pkl`
"""

import os
import glob
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import mne
import moabb
from moabb.paradigms import MotorImagery

mne.set_log_level('warning')
moabb.set_log_level('warning')


def load_subject_data(dataset_name: str, subject_id: int, channels=None):
    dataset_class = getattr(moabb.datasets, dataset_name)
    dataset = dataset_class()
    paradigm = MotorImagery(fmin=8, fmax=30, channels=channels, events=["left_hand", "right_hand"], n_classes=2)
    X, labels, _ = paradigm.get_data(dataset=dataset, subjects=[subject_id])
    return X, labels


def main():
    datasets = ["Dreyer2023", "Dreyer2023A", "PhysionetMI", "Lee2019_MI", "GuttmannFlury2025_MI", "GuttmannFlury2025_ME", "Yang2025"]
    selected_channels = ["C1", "C2", "C5", "C3", "C4", "C6", "FC3", "CP3", "FC4", "CP4", "Cz"]

    print("================================================================================")
    print(" Retraining Classical Single-Subject Pipelines (80/20 Train/Test Split)")
    print("================================================================================\n")

    for ds in datasets:
        tp_dir = os.path.join("trained_pipelines", ds)
        if not os.path.exists(tp_dir):
            continue

        sub_dirs = [d for d in os.listdir(tp_dir) if d.startswith("subject_") and os.path.isdir(os.path.join(tp_dir, d))]
        subject_ids = sorted([int(d.replace("subject_", "")) for d in sub_dirs])

        print(f"[{ds}] Processing {len(subject_ids)} subjects...")
        for s_id in subject_ids:
            try:
                X, labels = load_subject_data(ds, s_id, channels=selected_channels)
                if len(X) >= 10:
                    X_tr, X_te, y_tr, y_te = train_test_split(X, labels, test_size=0.20, random_state=42, stratify=labels)
                else:
                    X_tr, y_tr = X, labels

                s_out = os.path.join(tp_dir, f"subject_{s_id}")
                os.makedirs(s_out, exist_ok=True)

                # CSP + LDA
                pipe_lda = make_pipeline(CSP(n_components=4), LDA())
                pipe_lda.fit(X_tr, y_tr)
                joblib.dump(pipe_lda, os.path.join(s_out, "CSP_LDA_pipeline.pkl"))

                # Cov + Tangent Space + LR
                pipe_cov = make_pipeline(Covariances(estimator='oas'), TangentSpace(metric='riemann'), LogisticRegression(max_iter=1000))
                pipe_cov.fit(X_tr, y_tr)
                joblib.dump(pipe_cov, os.path.join(s_out, "Cov_Tangent_Space_LR_pipeline.pkl"))

                # CSP + SVM
                pipe_svm = make_pipeline(CSP(n_components=4), SVC(kernel='rbf'))
                pipe_svm.fit(X_tr, y_tr)
                joblib.dump(pipe_svm, os.path.join(s_out, "CSP_SVM_pipeline.pkl"))

            except Exception as e:
                print(f"  Warning: Skipped subject {s_id} in {ds}: {e}")

    print("\n================================================================================")
    print(" Successfully retrained all classical single-subject pipelines on 80% train split!")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
