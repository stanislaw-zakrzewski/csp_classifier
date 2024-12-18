import numpy as np

from data_classes.subject import Subject
from tkinter import filedialog as fd
from parafac_analysis.processing.parafac_analysis_spectral_smart import adapt_selected_channels, process

available_montages = ['biosemi64', 'standard_1020']


def main():
    subject_path = fd.askopenfilename(filetypes=[("European Data Format files", "*.edf")])
    subject_name = subject_path.split('/')[-1][:-4]
    subject = Subject(subject_path)
    print("Fill PARAFAC decomposition parameters")
    starting_rank = int(input('Starting rank: '))
    end_rank = int(input('End rank: '))

    replicas = input('Replica count (default: 5): ')
    replicas = int(replicas) if replicas != '' else 5

    iterations = input('Optimizer iterations >=10 (default: 10): ')
    iterations = int(iterations) if iterations != '' else 10
    if iterations < 10:
        iterations = 10

    print('Available montages:')
    for montage_index, available_montage in enumerate(available_montages):
        print(f'\t{montage_index + 1}: {available_montage}')
    montage = input('Select montage:')
    montage = available_montages[int(montage) - 1]

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
    selected_channels = adapt_selected_channels(subject, selected_channels)

    print("Available labels:")
    unique_events, event_counts = np.unique(subject.events[:, 2], return_counts=True)
    cardinalities = dict(zip(unique_events, event_counts))
    for label_name in label_names:
        print('\t* {}: {} (cardinality: {})'.format(label_name, label_names[label_name], cardinalities[label_name]))
    selected_label_1 = int(input("Select first label:"))
    selected_label_2 = int(input("Select second label:"))
    selected_labels = {
        selected_label_1: label_names[selected_label_1],
        selected_label_2: label_names[selected_label_2]
    }

    save_file_name = 'significant_heatmaps/{}'.format(subject_name)

    process(subject, selected_frequency_band, selected_channels, selected_labels, starting_rank, end_rank,
            replicas, iterations, t_min, t_max, montage, save_file_name=save_file_name, verbose='ERROR')

    print(f'Results saved to: {save_file_name}.npy')


main()
