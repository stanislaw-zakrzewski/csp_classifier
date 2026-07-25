import os
import pandas as pd
import numpy as np
import joblib

import moabb
from moabb.paradigms import MotorImagery
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
import mne
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Suppress verbose MNE and MOABB log output
mne.set_log_level('warning')
moabb.set_log_level('warning')

def train_and_evaluate_moabb(dataset, subject_list, bands, selected_channels, output_dir="trained_pipelines"):
    """
    Train and evaluate classifiers on a MOABB dataset for specific subjects.
    
    Args:
        dataset: A MOABB dataset instance (e.g., moabb.datasets.Cho2017()).
        subject_list: List of subject IDs to process.
        bands: List of frequency bands (e.g., [(8, 30)]).
        selected_channels: List of channels to select (e.g., ['C3', 'C4', 'Cz']).
        output_dir: Directory where trained pipelines and results will be saved.
    """
    dataset_name = dataset.code
    
    accuracies_data = []

    # Use the first frequency band provided
    fmin, fmax = bands[0]
    
    # MOABB MotorImagery paradigm to handle filtering and epoching automatically
    # MOABB standardizes event labels, so 'left_hand' and 'right_hand' will be correctly 
    # mapped to their dataset-specific equivalents automatically.
    paradigm = MotorImagery(
        fmin=fmin, 
        fmax=fmax, 
        channels=selected_channels if selected_channels else None,
        events=["left_hand", "right_hand"],
        n_classes=2
    )

    for subject_id in subject_list:
        print(f"Processing dataset {dataset_name}, Subject {subject_id}...")
        
        subject_output_dir = os.path.join(output_dir, dataset_name, f"subject_{subject_id}")
        os.makedirs(subject_output_dir, exist_ok=True)
        
        # Get epochs and labels for the current subject
        try:
            X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[subject_id])
        except Exception as e:
            print(f"  Failed to extract data for subject {subject_id}: {e}")
            continue
            
        pipelines = {
            "CSP + LDA": make_pipeline(CSP(n_components=4), LDA()),
            "Cov + Tangent Space + LR": make_pipeline(Covariances(estimator='oas'), TangentSpace(metric='riemann'), LogisticRegression(max_iter=1000)),
            "CSP + SVM": make_pipeline(CSP(n_components=4), SVC(kernel='rbf'))
        }
        
        subject_accuracies = {'dataset': dataset_name, 'subject': subject_id}
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for clf_name, pipeline in pipelines.items():
            print(f"  Training {clf_name}...")
            try:
                # Calculate 5-fold CV accuracy
                scores = cross_val_score(pipeline, X, labels, cv=cv, n_jobs=1)
                accuracy = np.mean(scores)
                subject_accuracies[clf_name] = accuracy
                
                # Fit the pipeline on the whole subject data and save it
                pipeline.fit(X, labels)
                
                pipeline_path = os.path.join(subject_output_dir, f"{clf_name.replace(' + ', '_').replace(' ', '_')}_pipeline.pkl")
                joblib.dump(pipeline, pipeline_path)
                print(f"    Saved {clf_name} pipeline with accuracy {accuracy:.4f}")
                
            except Exception as e:
                print(f"    Failed to train {clf_name}: {e}")
                subject_accuracies[clf_name] = None
                
        accuracies_data.append(subject_accuracies)
        
    # Save accuracies to CSV
    if accuracies_data:
        df_accuracies = pd.DataFrame(accuracies_data)
        csv_path = os.path.join(output_dir, dataset_name, "accuracies.csv")
        df_accuracies.to_csv(csv_path, index=False)
        print(f"\nSaved overall accuracies to {csv_path}")
        print(df_accuracies)

    return accuracies_data

