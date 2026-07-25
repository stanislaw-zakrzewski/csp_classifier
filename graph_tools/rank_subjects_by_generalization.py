"""
Subject Generalization Ranker
=============================

This script analyzes cross-subject directed performance graphs (.graphml) created from EEG classification simulations.
It ranks subjects based on how effectively a classifier trained on each subject's data generalizes to other target subjects.

Supports both standard graphs (in graph_results/graphs) and temporal binned graphs (in graph_results/graphs_binned).
"""

import os
import sys

# Re-export and delegate to top-level ranker if needed or standalone module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rank_subjects_by_generalization import (
    find_graphml_files,
    extract_graph_subject_stats,
    rank_subjects_by_generalization,
    main
)

if __name__ == "__main__":
    main()
