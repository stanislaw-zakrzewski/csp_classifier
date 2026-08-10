"""
Longitudinal Dataset Loader & Preprocessing Pipeline
=====================================================

Loads, cleans, standardizes channel names, resamples, filters, and epochs EDF files
from `longitudal_dataset/` for Experiment 10.

Features:
- Automatic session directory discovery (session1, session2, ..., sessionN).
- Channel name normalization (e.g. CZ -> Cz, FCZ -> FCz, etc.).
- Selection of 11 C-line sensimotor channels consistent with Exp 0-9.
- Resampling to target sfreq (default: 128 Hz or 250 Hz).
- Epoching (0.0 to 3.0s after trial onset).
- Label encoding: 'left' -> 0, 'right' -> 1.
- Splitting into training sessions (1..k) and test sessions (k+1..N).
"""

import os
import glob
import re
import mne
import numpy as np

# Suppress verbose MNE outputs
mne.set_log_level('WARNING')

# Standard 11 motor channel set used across MOABB experiments
TARGET_MOTOR_CHANNELS = [
    'FC3', 'FC1', 'Cz', 'FC2', 'FC4',
    'C3', 'C1', 'C2', 'C4',
    'CP3', 'CP4'
]

# Standard fallback 11 channels if exact set differs
FALLBACK_11_CHANNELS = [
    'C1', 'C2', 'C5', 'C3', 'C4', 'C6', 'FC3', 'CP3', 'FC4', 'CP4', 'Cz'
]


def normalize_channel_names(ch_names: list[str]) -> dict[str, str]:
    """
    Map uppercase/non-standard channel names to standard MNE mixed-case convention.
    e.g. 'CZ' -> 'Cz', 'FCZ' -> 'FCz', 'CPZ' -> 'CPz', 'FZ' -> 'Fz', 'PZ' -> 'Pz', 'POZ' -> 'POz'.
    """
    rename_dict = {}
    for ch in ch_names:
        clean_ch = ch.strip()
        # Handle 'Z' suffix to lower-case 'z'
        if clean_ch.endswith('Z') and len(clean_ch) <= 4:
            new_ch = clean_ch[:-1] + 'z'
            rename_dict[ch] = new_ch
        else:
            rename_dict[ch] = clean_ch
    return rename_dict


def get_available_sessions(data_dir: str = "longitudal_dataset") -> list[str]:
    """Find and sort all session folders (session1, session2, ...) in data_dir."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Longitudinal dataset directory '{data_dir}' not found.")
    
    sessions = []
    for entry in os.listdir(data_dir):
        full_path = os.path.join(data_dir, entry)
        if os.path.isdir(full_path) and entry.lower().startswith("session"):
            sessions.append(entry)
    
    # Sort numerically by session number (e.g. session1, session2, session10)
    def extract_session_num(name: str) -> int:
        match = re.search(r'\d+', name)
        return int(match.group()) if match else 999
        
    sessions.sort(key=extract_session_num)
    return sessions


def load_edf_file(
    file_path: str,
    target_sfreq: float = 128.0,
    fmin: float = 8.0,
    fmax: float = 30.0,
    tmin: float = 0.0,
    tmax: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a single EDF file, normalize channels, filter, resample, epoch and return (X, y).
    X shape: (n_trials, n_channels, n_samples)
    y shape: (n_trials,)
    """
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    
    # 1. Normalize channel names
    rename_map = normalize_channel_names(raw.ch_names)
    raw.rename_channels(rename_map)
    
    # 2. Filter 8-30 Hz
    raw.filter(l_freq=fmin, h_freq=fmax, method='iir', verbose=False)
    
    # 3. Channel selection
    available_chans = raw.ch_names
    selected_chans = [c for c in TARGET_MOTOR_CHANNELS if c in available_chans]
    if len(selected_chans) < 5:
        selected_chans = [c for c in FALLBACK_11_CHANNELS if c in available_chans]
    if len(selected_chans) == 0:
        # Fallback to all EEG channels if none matched
        selected_chans = available_chans[:11]
        
    raw.pick_channels(selected_chans)
    
    # 4. Extract annotations / events
    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    
    # Map event names to 0 / 1
    label_map = {}
    for key, val in event_dict.items():
        key_str = str(key).lower()
        if 'left' in key_str:
            label_map[val] = 0
        elif 'right' in key_str:
            label_map[val] = 1
        else:
            # Fallback sequential assignment for unknown labels
            label_map[val] = 0 if len(label_map) == 0 else 1
            
    # 5. Epoching
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_dict,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False
    )
    
    # 6. Resample epochs
    if epochs.info['sfreq'] != target_sfreq:
        epochs.resample(target_sfreq, verbose=False)
        
    X = epochs.get_data()  # (n_trials, n_chans, n_samples)
    raw_y = epochs.events[:, -1]
    y = np.array([label_map.get(lbl, 0) for lbl in raw_y], dtype=np.int64)
    
    return X, y


def load_session_data(
    session_name: str,
    modality: str = "DRY",  # "DRY", "WET", or "BOTH"
    data_dir: str = "longitudal_dataset",
    target_sfreq: float = 128.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all trials for a given session and modality.
    modality: "DRY", "WET", or "BOTH"
    """
    session_path = os.path.join(data_dir, session_name)
    if not os.path.isdir(session_path):
        raise FileNotFoundError(f"Session path '{session_path}' does not exist.")
        
    files = sorted(os.listdir(session_path))
    edf_files = [f for f in files if f.endswith(".edf")]
    
    if modality == "DRY":
        target_files = [f for f in edf_files if "_DRY" in f.upper()]
    elif modality == "WET":
        target_files = [f for f in edf_files if "_WET" in f.upper()]
    elif modality == "BOTH":
        target_files = edf_files
    else:
        raise ValueError(f"Unknown modality '{modality}'. Expected 'DRY', 'WET', or 'BOTH'.")
        
    if not target_files:
        raise FileNotFoundError(f"No EDF files found for modality '{modality}' in {session_path}")
        
    X_list, y_list = [], []
    for fname in target_files:
        fpath = os.path.join(session_path, fname)
        X_file, y_file = load_edf_file(fpath, target_sfreq=target_sfreq)
        X_list.append(X_file)
        y_list.append(y_file)
        
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y


def get_k_session_split(
    k: int,
    train_modality: str = "DRY",
    test_modality: str = "DRY",
    data_dir: str = "longitudal_dataset",
    target_sfreq: float = 128.0
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """
    Split available sessions into:
    - Training set: Concatenation of sessions 1..k (using train_modality).
    - Test dict: Dict mapping session_name -> (X_test, y_test) for sessions k+1..N (using test_modality).
    
    Returns:
        X_train, y_train, test_sessions_dict
    """
    sessions = get_available_sessions(data_dir)
    n_sessions = len(sessions)
    if k < 1 or k >= n_sessions:
        raise ValueError(f"k must be between 1 and {n_sessions - 1}, got k={k}")
        
    train_sessions = sessions[:k]
    test_sessions = sessions[k:]
    
    # Load training sessions
    X_tr_list, y_tr_list = [], []
    for sess in train_sessions:
        X_s, y_s = load_session_data(sess, modality=train_modality, data_dir=data_dir, target_sfreq=target_sfreq)
        X_tr_list.append(X_s)
        y_tr_list.append(y_s)
        
    X_train = np.concatenate(X_tr_list, axis=0)
    y_train = np.concatenate(y_tr_list, axis=0)
    
    # Load test sessions individually
    test_dict = {}
    for sess in test_sessions:
        X_te, y_te = load_session_data(sess, modality=test_modality, data_dir=data_dir, target_sfreq=target_sfreq)
        test_dict[sess] = (X_te, y_te)
        
    return X_train, y_train, test_dict
