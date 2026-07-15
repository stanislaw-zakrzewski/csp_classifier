from binascii import Error
import os
import joblib
import pandas as pd
import numpy as np
import moabb
from moabb.paradigms import MotorImagery
import copy
from sklearn.pipeline import make_pipeline
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def simulate_adaptive_classification(classifier_paths, dataset_name, subject_id, fmin=8, fmax=30, channels=None):
    """
    Simulates adaptive real-time classification for given pretrained classifiers on a specific MOABB subject.
    
    Args:
        classifier_paths (list of str): Paths to the pickled pipeline files.
        dataset_name (str): Name of the MOABB dataset (e.g. 'Cho2017').
        subject_id (int): Subject number from the MOABB dataset to test on.
        fmin (float): Minimum frequency for bandpass filter.
        fmax (float): Maximum frequency for bandpass filter.
        channels (list of str): Specific channels to use.
    """
    print(f"Loading dataset {dataset_name} for subject {subject_id}...")
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
    
    # Load classifiers
    classifiers = {}
    for path in classifier_paths:
        try:
            # We use pickle / joblib to load the model
            clf = joblib.load(path)
            # Create a unique name in case multiple subjects have same model name (e.g. CSP_LDA_pipeline.pkl)
            subject_folder = os.path.basename(os.path.dirname(path))
            clf_name = f"{subject_folder}_{os.path.basename(path).replace('.pkl', '')}"
            classifiers[clf_name] = clf
            print(f"Loaded {clf_name} from {path}")
        except Exception as e:
            print(f"Failed to load classifier from {path}: {e}")
            
    if not classifiers:
        print("No classifiers loaded successfully.")
        return None
        
    # Add baseline classifiers
    classifiers["baseline_CSP_LDA"] = make_pipeline(CSP(n_components=4), LDA())
    classifiers["baseline_Cov_Tangent_LR"] = make_pipeline(Covariances(estimator='oas'), TangentSpace(metric='riemann'), LogisticRegression(max_iter=1000))
    classifiers["baseline_CSP_SVM"] = make_pipeline(CSP(n_components=4), SVC(kernel='rbf'))
        
    unique_labels = np.unique(labels)
    
    # Add static copies of all pretrained classifiers (exclude baseline)
    static_classifiers = {}
    for clf_name, clf in classifiers.items():
        if not clf_name.startswith("baseline_"):
            static_classifiers[f"{clf_name}_static"] = copy.deepcopy(clf)
    classifiers.update(static_classifiers)
    
    # Iterate through classifiers
    for clf_name, clf in classifiers.items():
        is_adaptive = not clf_name.endswith("_static")
        print(f"Simulating {'adaptive' if is_adaptive else 'static'} classification for {clf_name}...")
        
        # Deep copy to not mutate the original loaded classifier if it's run multiple times
        pipeline = copy.deepcopy(clf)
        
        X_history = []
        y_history = []
        
        correct_predictions = 0
        total_predictions = 0
        
        for i, (x_trial, y_true) in enumerate(zip(X, labels)):
            # x_trial has shape (channels, samples)
            # Expand dims to (1, channels, samples) as expected by pipeline predict
            X_test = np.expand_dims(x_trial, axis=0)
            
            # Predict
            try:
                pred = pipeline.predict(X_test)[0]
            except Exception as e:
                # Ignore prediction errors (e.g., NotFittedError for baseline models early on)
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
            
            # Append to history
            X_history.append(x_trial)
            y_history.append(y_true)
            
            # Update classifier
            if is_adaptive and len(X_history) >= 4 and len(X_history) % 4 == 0:
                # Ensure we have at least one of each active class
                if len(np.unique(y_history)) == len(unique_labels):
                    try:
                        X_arr = np.array(X_history)
                        y_arr = np.array(y_history)
                        if hasattr(pipeline, 'partial_fit'):
                            pipeline.partial_fit(X_arr, y_arr)
                        else:
                            pipeline.fit(X_arr, y_arr)
                    except Exception as e:
                        print(f"Failed to fit on trial {i}: {e}")
                        pass
                        
    # Save results
    df_results = pd.DataFrame(results)
    output_filename = f"simulation_results/{dataset_name}/{subject_id}.csv"
    df_results.to_csv(output_filename, index=False)
    print(f"Results saved to {output_filename}")
    
    return df_results

if __name__ == "__main__":
    DATASET_NAME = "Cho2017"
    DATASET_NAME = "PhysionetMI"
    DATASET_NAME = "Dreyer2023"

    # Script to run process for pretrained classifiers for first 10 subjects from trained pipelines
    # and run them for subject no 14 from Cho2017.
    
    # Selected channels from run_cho2017.py
    selected_channels = ["C1", "C2", "C5", "C3", "C4", "C6", "FC3", "CP3", "FC4", "CP4", "Cz"]
    
    # Get all pipeline paths for the first 10 subjects
    trained_pipelines_dir = os.path.join("trained_pipelines", DATASET_NAME)
    
    # Load accuracies to filter classifiers by a threshold
    accuracies_csv_path = os.path.join(trained_pipelines_dir, "accuracies.csv")
    accuracies_df = pd.DataFrame()
    if os.path.exists(accuracies_csv_path):
        accuracies_df = pd.read_csv(accuracies_csv_path)
        
    ACCURACY_THRESHOLD = 0.2  # Threshold for pretrained classifier accuracy
    
    def get_column_from_filename(filename):
        if "CSP_LDA" in filename:
            return "CSP + LDA"
        elif "Cov_Tangent_Space_LR" in filename:
            return "Cov + Tangent Space + LR"
        elif "CSP_SVM" in filename:
            return "CSP + SVM"
        return None

    # Determine the number of subjects in the selected trained pipelines directory
    subject_dirs = [d for d in os.listdir(trained_pipelines_dir) 
                    if d.startswith("subject_") and os.path.isdir(os.path.join(trained_pipelines_dir, d))]
    num_subjects = len(subject_dirs)

    for focused_subject in range(1, 2):# num_subjects + 1):
        classifier_paths = []
    
        # Loop through all subjects
        for sub in range(1, num_subjects + 1):
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
                        
        print(f"Found {len(classifier_paths)} pretrained classifiers meeting threshold for subject {focused_subject}.")
        
        if len(classifier_paths) > 0:
            # Run them for subject no 14 from Cho2017
            simulate_adaptive_classification(
                classifier_paths=classifier_paths,
                dataset_name=DATASET_NAME,
                subject_id=focused_subject,
                fmin=8,
                fmax=30,
                channels=selected_channels
            )
        else:
            print("No classifiers found to simulate. Please run train_moabb.py / run_cho2017.py first.")
