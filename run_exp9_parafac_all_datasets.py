"""
Experiment 9: Batch MOABB PARAFAC Heatmap Generation
====================================================

Iterates through all 7 MOABB datasets (423 subjects total), extracts raw signals and events,
runs IRASA spectral decomposition and PARAFAC non-negative tensor factorization (250 Hz, 11 MI channels),
and saves subject significance heatmaps into `graph_results/parafac_diagnostics/{dataset}/`.
"""

import os
import sys
import argparse
import numpy as np
import mne
from mne import get_config
from moabb.datasets import (
    Dreyer2023,
    Dreyer2023A,
    PhysionetMI,
    Lee2019_MI,
    GuttmannFlury2025_MI,
    GuttmannFlury2025_ME,
    Yang2025
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from parafac_analysis.processing.parafac_moabb_adapter import process_moabb_raw, DEFAULT_CHANNELS


DATASET_MAP = {
    "Dreyer2023": Dreyer2023,
    "Dreyer2023A": Dreyer2023A,
    "PhysionetMI": PhysionetMI,
    "Lee2019_MI": Lee2019_MI,
    "GuttmannFlury2025_MI": GuttmannFlury2025_MI,
    "GuttmannFlury2025_ME": GuttmannFlury2025_ME,
    "Yang2025": Yang2025
}


def process_dataset(dataset_name, max_subjects=None, overwrite=False, out_base_dir="graph_results/parafac_diagnostics", use_fallback=True):
    if dataset_name not in DATASET_MAP:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"\n================================================================================")
    print(f" Processing PARAFAC Factorization for Dataset: {dataset_name} (Fallback={use_fallback})")
    print(f" Output Dir: {out_base_dir}")
    print(f"================================================================================\n")

    dataset_cls = DATASET_MAP[dataset_name]
    dataset = dataset_cls()
    subjects = dataset.subject_list

    if max_subjects is not None:
        subjects = subjects[:max_subjects]

    output_dir = os.path.join(out_base_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    for sub_id in subjects:
        save_path = os.path.join(output_dir, f"subject_{sub_id}_heatmap.npy")
        if os.path.exists(save_path) and not overwrite:
            print(f"[{dataset_name}] Subject {sub_id} heatmap already exists. Skipping.")
            continue

        try:
            print(f"[{dataset_name}] Loading raw data for Subject {sub_id}...")
            data = dataset.get_data(subjects=[sub_id])
            subject_data = data[sub_id]

            # Concatenate sessions/runs into single raw & events
            raw_list = []
            for sess_name, sess_data in subject_data.items():
                for run_name, raw in sess_data.items():
                    raw_list.append(raw)

            if len(raw_list) == 1:
                raw_combined = raw_list[0]
            else:
                raw_combined = mne.concatenate_raws(raw_list, verbose=False)

            events, event_id = mne.events_from_annotations(raw_combined, verbose=False)

            # Keep 2 main MI classes
            if len(event_id) > 2:
                valid_keys = list(event_id.keys())[:2]
                id_dict = {k: event_id[k] for k in valid_keys}
            else:
                id_dict = event_id

            print(f"[{dataset_name}] Running PARAFAC decomposition for Subject {sub_id} (Channels: 11 MI, SFreq: 250 Hz)...")
            payload = process_moabb_raw(
                raw_signal=raw_combined,
                events=events,
                id_dict=id_dict,
                subject_id=sub_id,
                dataset_name=dataset_name,
                output_dir=output_dir,
                band=(8, 30),
                selected_channels=DEFAULT_CHANNELS,
                starting_rank=5,
                end_rank=10,
                replicas=3,
                iterations=5,
                t_min=0.0,
                t_max=3.0,
                resample_freq=250,
                use_fallback=use_fallback,
                verbose="ERROR"
            )
            if payload is not None:
                print(f"  --> Saved: {save_path}")

        except Exception as e:
            print(f"ERROR processing Subject {sub_id} in {dataset_name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 9 PARAFAC Decomposition across MOABB datasets.")
    parser.add_argument("--dataset", type=str, default="all", help="Target dataset name or 'all'.")
    parser.add_argument("--max_subjects", type=int, default=None, help="Limit number of subjects per dataset.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing heatmap npy files.")
    parser.add_argument("--out_dir", type=str, default="graph_results/parafac_diagnostics", help="Output directory base.")
    parser.add_argument("--no_fallback", action="store_true", help="Disable top 20% fallback to enforce strict p <= 0.10 testing.")
    args = parser.parse_args()

    use_fallback = not args.no_fallback
    mne.set_log_level("ERROR")

    if args.dataset == "all":
        for ds_name in DATASET_MAP.keys():
            process_dataset(ds_name, max_subjects=args.max_subjects, overwrite=args.overwrite, out_base_dir=args.out_dir, use_fallback=use_fallback)
    else:
        process_dataset(args.dataset, max_subjects=args.max_subjects, overwrite=args.overwrite, out_base_dir=args.out_dir, use_fallback=use_fallback)

    print("\n================================================================================")
    print(" PARAFAC HEATMAP GENERATION COMPLETE FOR ALL TARGET DATASETS")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
