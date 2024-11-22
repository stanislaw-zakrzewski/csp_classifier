import traceback

import numpy as np
import os

from data_classes.subject import Subject
from tkinter import filedialog as fd

from parafac_analysis.processing.parafac_analysis_spectral_smart import adapt_selected_channels, process

from filter_edf import main as create_tmp

stage = 1


def main():
    folder_path = fd.askdirectory()
    # folder_path = 'dataset_temp'
    print("Fill PARAFAC decomposition parameters")
    starting_rank = int(input('Starting rank: '))
    end_rank = int(input('End rank: '))

    replicas = input('Replica count (default: 5): ')
    replicas = int(replicas) if replicas != '' else 5

    iterations = input('Optimizer iterations >=10 (default: 10): ')
    iterations = int(iterations) if iterations != '' else 10

    selected_frequency_band = input('Selected frequency band (default: "2,28"): ')
    if selected_frequency_band:
        selected_frequency_band = selected_frequency_band.split(',')
        selected_frequency_band = (int(selected_frequency_band[0]), int(selected_frequency_band[1]))
    else:
        selected_frequency_band = (2, 28)
    t_min = input('Time after cue to start cut (default: 0): ')
    t_min = float(t_min) if t_min != '' else .0

    t_max = input('Time after cue to end cut (default: 1): ')
    t_max = float(t_max) if t_max != '' else 1.0

    selected_channels = None
    selected_label_1_name = None
    selected_label_2_name = None
    selected_labels = None

    for file in os.listdir(folder_path):
        file_name = os.fsdecode(file)
        try:
            if file_name.endswith(".edf"):
                subject_name = file_name[:-4]
                print(file_name)
                subject = Subject("{}/{}".format(folder_path, file_name))
                if stage == 2:
                    heatmap = np.load("parafac_analysis/significant_heatmaps/{}.npy".format(subject_name),
                                      allow_pickle=True).item()
                    create_tmp(heatmap, subject)
                    subject = Subject('filtered.edf')

                if selected_channels is None:
                    print('Available channels: {}'.format(', '.join(subject.electrode_names)))
                    selected_channels = input('Select channels (default: "C5,C3,C1,Cz,C2,C4,C6", all: "ALL"): ')
                    if selected_channels == 'ALL':
                        selected_channels = subject.electrode_names
                    elif selected_channels:
                        selected_channels = selected_channels.split(',')
                    else:
                        selected_channels = ['C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6']
                    selected_channels = adapt_selected_channels(subject, selected_channels)

                label_names = {}
                for label_name in subject.id_dict:
                    label_names[subject.id_dict[label_name]] = label_name

                unique_events, event_counts = np.unique(subject.events[:, 2], return_counts=True)
                cardinalities = dict(zip(unique_events, event_counts))
                if selected_label_1_name is None or selected_label_2_name is None:
                    print("Available labels:")

                    for label_name in label_names:
                        print('\t* {}: {} (cardinality: {})'.format(label_name, label_names[label_name],
                                                                    cardinalities[label_name]))
                    selected_label_1 = int(input("Select first label:"))
                    selected_label_2 = int(input("Select second label:"))
                    selected_label_1_name = label_names[selected_label_1]
                    selected_label_2_name = label_names[selected_label_2]
                    selected_labels = {
                        selected_label_1: selected_label_1_name,
                        selected_label_2: selected_label_2_name
                    }
                else:
                    new_key_label_1 = list(label_names.keys())[list(label_names.values()).index(selected_label_1_name)]
                    new_key_label_2 = list(label_names.keys())[list(label_names.values()).index(selected_label_2_name)]
                    selected_labels = {
                        new_key_label_1: selected_label_1_name,
                        new_key_label_2: selected_label_2_name,
                    }

                if stage == 1:
                    save_file_name = 'significant_heatmaps/{}'.format(subject_name)
                else:
                    save_file_name = 'significant_heatmaps_stage_2/{}'.format(subject_name)

                process(subject, selected_frequency_band, selected_channels, selected_labels, starting_rank, end_rank,
                        replicas, iterations, t_min, t_max, save_file_name=save_file_name, verbose='ERROR')
        except Exception as e:
            print(traceback.format_exc())


main()
