"""
Simulate Missing Dreyer2023 Subjects (Subjects 14 and 15)
==========================================================

This script runs cross-subject adaptive and static simulations for the newly trained pipelines from Subject 14 and Subject 15 across all target subjects (1 to 87) in the Dreyer2023 dataset.

It carefully APPENDS the new simulation results to existing CSV files in simulation_results/Dreyer2023/ without overwriting previous results or duplicating entries.
"""

import os
import joblib
import pandas as pd
import numpy as np
import moabb
from moabb.paradigms import MotorImagery
import copy
from sklearn.pipeline import make_pipeline
import mne
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from tqdm import tqdm

# Suppress verbose MNE and MOABB output
mne.set_log_level('warning')
moabb.set_log_level('warning')


def simulate_and_append_classifiers(
    classifier_paths,
    dataset_name,
    subject_id,
    fmin=8,
    fmax=30,
    channels=None,
    include_baselines_if_missing=True
):
    """
    Simulates adaptive and static classification for given pretrained classifiers on a specific subject,
    and appends the results to the existing CSV file for that subject.
    """
    output_dir = os.path.join("simulation_results", dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"{subject_id}.csv")
    csv_exists = os.path.exists(output_filename)

    try:
        dataset_class = getattr(moabb.datasets, dataset_name)
        dataset = dataset_class()
    except AttributeError:
        print(f"Dataset {dataset_name} not found in moabb.datasets")
        return None

    paradigm = MotorImagery(
        fmin=fmin,
        fmax=fmax,
        channels=channels,
        events=["left_hand", "right_hand"],
        n_classes=2
    )

    try:
        X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[subject_id])
    except Exception as e:
        print(f"Failed to get data for subject {subject_id}: {e}")
        return None

    results = []

    # Load pretrained classifiers
    classifiers = {}
    for path in classifier_paths:
        try:
            clf = joblib.load(path)
            subject_folder = os.path.basename(os.path.dirname(path))
            clf_name = f"{subject_folder}_{os.path.basename(path).replace('.pkl', '')}"
            classifiers[clf_name] = clf
        except Exception as e:
            print(f"Failed to load classifier from {path}: {e}")

    if not classifiers:
        print(f"No classifiers loaded for target subject {subject_id}.")
        return None

    # Include baselines ONLY if the target CSV doesn't exist yet
    if not csv_exists and include_baselines_if_missing:
        classifiers["baseline_CSP_LDA"] = make_pipeline(CSP(n_components=4), LDA())
        classifiers["baseline_Cov_Tangent_LR"] = make_pipeline(
            Covariances(estimator='oas'), TangentSpace(metric='riemann'), LogisticRegression(max_iter=1000)
        )
        classifiers["baseline_CSP_SVM"] = make_pipeline(CSP(n_components=4), SVC(kernel='rbf'))

    unique_labels = np.unique(labels)

    # Add static copies of all pretrained classifiers
    static_classifiers = {}
    for clf_name, clf in classifiers.items():
        if not clf_name.startswith("baseline_"):
            static_classifiers[f"{clf_name}_static"] = copy.deepcopy(clf)
    classifiers.update(static_classifiers)

    # Iterate through classifiers and run simulation over trials
    for clf_name, clf in classifiers.items():
        is_adaptive = not clf_name.endswith("_static")

        pipeline = copy.deepcopy(clf)
        X_history = []
        y_history = []

        correct_predictions = 0
        total_predictions = 0

        for i, (x_trial, y_true) in enumerate(zip(X, labels)):
            X_test = np.expand_dims(x_trial, axis=0)

            try:
                pred = pipeline.predict(X_test)[0]
            except Exception:
                pred = None

            is_correct = (str(pred).lower() == str(y_true).lower())
            if is_correct:
                correct_predictions += 1
            total_predictions += 1

            results.append({
                'Classifier': clf_name,
                'Trial': i,
                'True_Label': y_true,
                'Predicted_Label': pred,
                'Is_Correct': is_correct,
                'Cumulative_Accuracy': correct_predictions / total_predictions
            })

            X_history.append(x_trial)
            y_history.append(y_true)

            if is_adaptive and len(X_history) >= 20 and len(X_history) % 20 == 0:
                if len(np.unique(y_history)) == len(unique_labels):
                    try:
                        X_arr = np.array(X_history)
                        y_arr = np.array(y_history)
                        if hasattr(pipeline, 'partial_fit'):
                            pipeline.partial_fit(X_arr, y_arr)
                        else:
                            pipeline.fit(X_arr, y_arr)
                    except Exception:
                        pass

    df_new = pd.DataFrame(results)

    # APPEND LOGIC: Read existing CSV, replace matching classifier rows if any, append and save
    if csv_exists:
        df_existing = pd.read_csv(output_filename)
        new_clf_names = set(df_new['Classifier'].unique())
        # Filter out any existing rows matching the new classifiers to prevent duplicate runs
        df_existing_clean = df_existing[~df_existing['Classifier'].isin(new_clf_names)]
        df_final = pd.concat([df_existing_clean, df_new], ignore_index=True)
    else:
        df_final = df_new

    df_final.to_csv(output_filename, index=False)
    print(f"Appended {len(df_new)} rows to {output_filename} (Total rows in CSV: {len(df_final)})")
    return df_final


def main():
    DATASET_NAME = "Dreyer2023"
    selected_channels = ["C1", "C2", "C5", "C3", "C4", "C6", "FC3", "CP3", "FC4", "CP4", "Cz"]
    trained_pipelines_dir = os.path.join("trained_pipelines", DATASET_NAME)

    accuracies_csv_path = os.path.join(trained_pipelines_dir, "accuracies.csv")
    accuracies_df = pd.DataFrame()
    if os.path.exists(accuracies_csv_path):
        accuracies_df = pd.read_csv(accuracies_csv_path)

    ACCURACY_THRESHOLD = 0.00001

    def get_column_from_filename(filename):
        if "CSP_LDA" in filename:
            return "CSP + LDA"
        elif "Cov_Tangent_Space_LR" in filename:
            return "Cov + Tangent Space + LR"
        elif "CSP_SVM" in filename:
            return "CSP + SVM"
        return None

    # Focus on subjects 14 and 15 as missing source subjects
    missing_source_subjects = [1]

    # Total subjects in dataset
    subject_dirs = [d for d in os.listdir(trained_pipelines_dir)
                    if d.startswith("subject_") and os.path.isdir(os.path.join(trained_pipelines_dir, d))]
    num_subjects = len(subject_dirs)

    print(f"=== Running Missing Simulations for Subjects 14 & 15 ({DATASET_NAME}) ===")
    print(f"Total dataset subjects found: {num_subjects}")

    # For each target subject (1..87)
    for focused_subject in range(1, num_subjects + 1):
        classifier_paths = []

        # Gather new pipelines from missing_source_subjects (14 and 15)
        for sub in missing_source_subjects:
            if sub == focused_subject:
                continue

            sub_dir = os.path.join(trained_pipelines_dir, f"subject_{sub}")
            if os.path.isdir(sub_dir):
                for file in os.listdir(sub_dir):
                    if file.endswith(".pkl"):
                        add_classifier = True
                        if not accuracies_df.empty:
                            col_name = get_column_from_filename(file)
                            if col_name:
                                acc_series = accuracies_df.loc[accuracies_df['subject'] == sub, col_name]
                                if not acc_series.empty:
                                    acc = acc_series.values[0]
                                    if acc < ACCURACY_THRESHOLD:
                                        add_classifier = False

                        if add_classifier:
                            classifier_paths.append(os.path.join(sub_dir, file))

        if len(classifier_paths) > 0:
            print(f"\nTarget Subject {focused_subject}: Simulating {len(classifier_paths)} pipelines from subjects 14 & 15...")
            simulate_and_append_classifiers(
                classifier_paths=classifier_paths,
                dataset_name=DATASET_NAME,
                subject_id=focused_subject,
                fmin=8,
                fmax=30,
                channels=selected_channels
            )
        else:
            print(f"\nTarget Subject {focused_subject}: No new classifier paths to add.")

    print("\n=== Missing simulation processing complete for Subjects 14 & 15 ===")


if __name__ == "__main__":
    main()
